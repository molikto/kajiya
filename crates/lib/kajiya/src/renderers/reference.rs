use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration, shader::ShaderSource},
};
use kajiya_rg::{self as rg};
use rg::{RenderGraph};

use crate::hit_groups::new_rt_with_default_hit_groups;

pub fn reference_path_trace(
    rg: &mut RenderGraph,
    output_img: &mut rg::Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
    tlas: &rg::Handle<RayTracingAcceleration>,
) {
    new_rt_with_default_hit_groups(
        rg.add_pass("reference pt"),
        ShaderSource::hlsl("/shaders/rt/reference_path_trace.rgen.hlsl"),
        [
            ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
            ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
        ],
        true
    )
    .write(output_img)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .trace_rays(tlas, output_img.desc().extent);
}
