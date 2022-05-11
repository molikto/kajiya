#include "inc/rt.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

struct RayHitAttrib
{
    float3 normal;
};

[shader("closesthit")]
void main(inout GbufferRayPayload payload: SV_RayPayload, in RayHitAttrib attrib: SV_IntersectionAttributes) {
    float3 normal = attrib.normal;
    GbufferData gbuffer = GbufferData::create_zero();
    gbuffer.albedo = float3(1.0, 0.0, 0.0);
    gbuffer.normal = normal;
    gbuffer.roughness = 0.0;
    gbuffer.metalness = 0.0;
    gbuffer.emissive = float3(0.0, 0.0, 0.0);
    payload.gbuffer_packed = gbuffer.pack();
    payload.t = RayTCurrent();
}
