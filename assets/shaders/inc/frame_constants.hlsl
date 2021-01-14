#ifndef FRAME_CONSTANTS_HLSL
#define FRAME_CONSTANTS_HLSL

#include "uv.hlsl"

struct ViewConstants {
    float4x4 view_to_clip;
    float4x4 clip_to_view;
    float4x4 view_to_sample;
    float4x4 sample_to_view;
    float4x4 world_to_view;
    float4x4 view_to_world;
    float2 sample_offset_pixels;
    float2 sample_offset_clip;
};

struct FrameConstants {
    ViewConstants view_constants;
    float4 mouse;
    uint frame_index;
};

[[vk::binding(0, 2)]] ConstantBuffer<FrameConstants> frame_constants;

struct ViewRayContext {
    float4 ray_dir_cs;
    float4 ray_dir_vs_h;
    float4 ray_dir_ws_h;

    float4 ray_origin_cs;
    float4 ray_origin_vs_h;
    float4 ray_origin_ws_h;

    float3 ray_dir_vs() {
        return ray_dir_vs_h.xyz;
    }

    float3 ray_dir_ws() {
        return ray_dir_ws_h.xyz;
    }

    float3 ray_origin_vs() {
        return ray_origin_vs_h.xyz / ray_origin_vs_h.w;
    }

    float3 ray_origin_ws() {
        return ray_origin_ws_h.xyz / ray_origin_ws_h.w;
    }

    static ViewRayContext from_uv(float2 uv) {
        ViewConstants view_constants = frame_constants.view_constants;

        ViewRayContext res;
        res.ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
        res.ray_dir_vs_h = mul(view_constants.sample_to_view, res.ray_dir_cs);
        res.ray_dir_ws_h = mul(view_constants.view_to_world, res.ray_dir_vs_h);

        res.ray_origin_cs = float4(uv_to_cs(uv), 1.0, 1.0);
        res.ray_origin_vs_h = mul(view_constants.sample_to_view, res.ray_origin_cs);
        res.ray_origin_ws_h = mul(view_constants.view_to_world, res.ray_origin_vs_h);

        return res;
    }
};


#endif