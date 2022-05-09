#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/bindless_textures.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] RWTexture2D<float3> geometric_normal_tex;
[[vk::binding(1)]] RWTexture2D<float4> gbuffer_tex;
[[vk::binding(2)]] RWTexture2D<float> depth_tex;
[[vk::binding(3)]] RWTexture2D<float3> velocity_tex;

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    float4 radiance_sample_count_packed = 0.0;

    static const uint sample_count = 1;
    const float2 pixel_center = px + float2(0.5, 0.5);
    const float2 uv = pixel_center / DispatchRaysDimensions().xy;

    RayDesc outgoing_ray;
    {
        const ViewRayContext view_ray_context = ViewRayContext::from_uv(uv);
        const float3 ray_dir_ws = view_ray_context.ray_dir_ws();

        outgoing_ray = new_ray(
            view_ray_context.ray_origin_ws(), 
            normalize(ray_dir_ws.xyz),
            0.0,
            FLT_MAX
        );
    }

    RayCone ray_cone = pixel_ray_cone_from_image_height(DispatchRaysDimensions().y);

    // Bias for texture sharpness
    ray_cone.spread_angle *= 0.3;

    GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
        .with_cone(ray_cone)
        .with_cull_back_faces(false)
        .with_path_length(0)
        .trace(acceleration_structure);

    if (!primary_hit.is_hit) {
        primary_hit.gbuffer_packed = GbufferData::create_zero().pack();
    }

    // TODO geometry normal instead of shading normal
    float3 geometric_normal_vs = direction_world_to_view(primary_hit.gbuffer_packed.unpack_normal()) * 0.5 + 0.5;

    gbuffer_tex[px] = asfloat(primary_hit.gbuffer_packed.data0);
    const float3 depth_vec = position_world_to_sample(primary_hit.position);
    depth_tex[px] = depth_vec.z;
    geometric_normal_tex[px] = geometric_normal_vs;
    // TODO velocity
    velocity_tex[px] = float3(0.0, 0.0, 0.0);
}
