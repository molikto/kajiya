#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"

#define USE_CSGI 1
#define USE_TEMPORAL_JITTER 1
#define USE_SHORT_RAYS_ONLY 1
#define SHORT_RAY_SIZE_VOXEL_CELLS 4.0
#define ROUGHNESS_BIAS 0.5
#define SUPPRESS_GI_FOR_NEAR_HITS 1

// Should be 1, but rarely matters for the diffuse bounce, so might as well save a few cycles.
#define USE_SOFT_SHADOWS 0

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] StructuredBuffer<uint> ranking_tile_buf;
[[vk::binding(3)]] StructuredBuffer<uint> scambling_tile_buf;
[[vk::binding(4)]] StructuredBuffer<uint> sobol_buf;
[[vk::binding(5)]] RWTexture2D<float4> out0_tex;
[[vk::binding(6)]] RWTexture2D<float4> out1_tex;
[[vk::binding(7)]] Texture3D<float4> csgi_direct_tex;
[[vk::binding(8)]] Texture3D<float4> csgi_indirect_tex;
[[vk::binding(9)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(10)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "../csgi/common.hlsl"
#include "../csgi/lookup.hlsl"


float blue_noise_sampler(int pixel_i, int pixel_j, int sampleIndex, int sampleDimension)
{
	// wrap arguments
	pixel_i = pixel_i & 127;
	pixel_j = pixel_j & 127;
	sampleIndex = sampleIndex & 255;
	sampleDimension = sampleDimension & 255;

	// xor index based on optimized ranking
	// jb: 1spp blue noise has all 0 in ranking_tile_buf so we can skip the load
	int rankedSampleIndex = sampleIndex ^ ranking_tile_buf[sampleDimension + (pixel_i + pixel_j*128)*8];

	// fetch value in sequence
	int value = sobol_buf[sampleDimension + rankedSampleIndex*256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scambling_tile_buf[(sampleDimension%8) + (pixel_i + pixel_j*128)*8];

	// convert to float and return
	float v = (0.5f+value)/256.0f;
	return v;
}

float3 sample_sun_direction(uint2 px) {
    #if USE_SOFT_SHADOWS
        if (frame_constants.sun_angular_radius_cos < 1.0) {
            const float3x3 basis = build_orthonormal_basis(normalize(SUN_DIRECTION));

            // 256x256 blue noise
            const uint noise_offset = frame_constants.frame_index;
            float2 urand = bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][
                (px + int2(noise_offset * 59, noise_offset * 37)) & 255
            ].xy * 255.0 / 256.0 + 0.5 / 256.0;

            return mul(basis, uniform_sample_cone(urand, frame_constants.sun_angular_radius_cos));
        }
    #endif

    return SUN_DIRECTION;
}


static const float SKY_DIST = 1e5;

[shader("raygeneration")]
void main() {
    uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const uint2 px = DispatchRaysIndex().xy;

    // TODO: jitter half-res buffers too; otherwise this is ineffective
    //const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
    const uint2 hi_px = px * 2;
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        out0_tex[px] = float4(0.0.xxx, -SKY_DIST);
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
    const float3 ray_dir_ws = normalize(view_ray_context.ray_dir_ws_h.xyz);
    float3 wo = mul(-ray_dir_ws, shading_basis);

    const float3 ray_hit_ws = view_ray_context.ray_hit_ws();
    const float3 ray_hit_vs = view_ray_context.ray_hit_vs();
    const float3 refl_ray_origin = ray_hit_ws - ray_dir_ws * (length(ray_hit_vs) + length(ray_hit_ws)) * 1e-4;

    const float3 primary_hit_normal = gbuffer.normal;

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    DiffuseBrdf brdf;
    brdf.albedo = 1.0.xxx;

    const uint seed = USE_TEMPORAL_JITTER ? frame_constants.frame_index : 0;
    uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), seed);

#if 1
    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);

    float2 urand = float2(
        blue_noise_sampler(px.x, px.y, noise_offset, 0),
        blue_noise_sampler(px.x, px.y, noise_offset, 1)
    );
#elif 0
    // 256x256 blue noise

    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);

    float2 urand = bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][
        (px + int2(noise_offset * 59, noise_offset * 37)) & 255
    ].xy * 255.0 / 256.0 + 0.5 / 256.0;
#elif 0
    float2 urand = float2(
        uint_to_u01_float(hash1_mut(rng)),
        uint_to_u01_float(hash1_mut(rng))
    );
#else
    float2 urand = frac(
        hammersley((frame_constants.frame_index * 5) % 16, 16) +
        bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][px & 255].xy * 255.0 / 256.0 + 0.5 / 256.0
    );
#endif

    BrdfSample brdf_sample = brdf.sample(wo, urand);

    float3 control_variate = 0.0.xxx;
    float3 total_radiance = 0.0.xxx;

    if (brdf_sample.is_valid()) {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
        outgoing_ray.Origin = refl_ray_origin;
        outgoing_ray.TMin = 0;

        #if USE_SHORT_RAYS_ONLY
            outgoing_ray.TMax = CSGI_VOXEL_SIZE.x * SHORT_RAY_SIZE_VOXEL_CELLS;
        #else
            outgoing_ray.TMax = SKY_DIST;
        #endif

        {
            float3 to_eye = get_eye_position() - ray_hit_ws;
            float3 pseudo_bent_normal = normalize(normalize(to_eye) + gbuffer.normal);

            control_variate = lookup_csgi(
                ray_hit_ws,
                gbuffer.normal,
                CsgiLookupParams::make_default()
                    .with_sample_directional_radiance(outgoing_ray.Direction)
                    .with_bent_normal(pseudo_bent_normal)
            );
        }

        out1_tex[px] = float4(outgoing_ray.Direction, clamp(brdf_sample.pdf, 1e-5, 1e5));

        // TODO: cone spread angle
        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray, 1.0);

        if (primary_hit.is_hit) {
            const float3 to_light_norm = sample_sun_direction(px);
            const bool is_shadowed =
                rt_is_shadowed(
                    acceleration_structure,
                    new_ray(
                        primary_hit.position,
                        to_light_norm,
                        1e-4,
                        SKY_DIST
                ));

            GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
            gbuffer.roughness = lerp(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
            const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
            const float3 wi = mul(to_light_norm, shading_basis);
            float3 wo = mul(-outgoing_ray.Direction, shading_basis);

            LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

            const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
            const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
            total_radiance += brdf_value * light_radiance;

            total_radiance += gbuffer.emissive;

            if (USE_CSGI) {
                const float3 pseudo_bent_normal = normalize(normalize(get_eye_position() - primary_hit.position) + gbuffer.normal);

                CsgiLookupParams lookup_params =
                    CsgiLookupParams::make_default()
                        .with_bent_normal(pseudo_bent_normal)
                        //.with_linear_fetch(false)
                        ;

                if (SUPPRESS_GI_FOR_NEAR_HITS && primary_hit.ray_t <= CSGI_VOXEL_SIZE.x) {
                    float max_normal_offset = primary_hit.ray_t * abs(dot(outgoing_ray.Direction, gbuffer.normal));

                    // Suppression in open corners causes excessive darkening,
                    // and doesn't prevent that many leaks. This strikes a balance.
                    const float normal_agreement = dot(primary_hit_normal, gbuffer.normal);
                    max_normal_offset = lerp(max_normal_offset, 1.51, normal_agreement * 0.5 + 0.5);

                    lookup_params = lookup_params
                        .with_max_normal_offset_scale(max_normal_offset / CSGI_VOXEL_SIZE.x);
                }

                float3 csgi = lookup_csgi(
                    primary_hit.position,
                    gbuffer.normal,
                    lookup_params
                );

                //if (primary_hit.ray_t > CSGI_VOXEL_SIZE.x)
                total_radiance += csgi * gbuffer.albedo;
            }
        } else {
            #if USE_SHORT_RAYS_ONLY
                const float3 far_gi = lookup_csgi(
                    outgoing_ray.Origin + outgoing_ray.Direction * max(0.0, outgoing_ray.TMax - CSGI_VOXEL_SIZE.x),
                    0.0.xxx,    // don't offset by any normal
                    CsgiLookupParams::make_default()
                        .with_sample_directional_radiance(outgoing_ray.Direction)
                );
            #else
                const float3 far_gi = sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
            #endif

            #if USE_RTDGI_CONTROL_VARIATES
                out0_tex[px] = float4(far_gi - control_variate, 1);
                //out0_tex[px] = float4(control_variate, 1);
            #else
                out0_tex[px] = float4(far_gi, 1);
            #endif
            //out0_tex[px] = float4(control_variate, 1);
            return;
        }

        #if USE_RTDGI_CONTROL_VARIATES
            float3 out_value = total_radiance - control_variate;
            //float3 out_value = control_variate;
        #else
            float3 out_value = total_radiance;
        #endif
        //float3 out_value = control_variate;

        out0_tex[px] = float4(out_value, 1);
    } else {
        out0_tex[px] = float4(0.0.xxx, 1);
    }
}
