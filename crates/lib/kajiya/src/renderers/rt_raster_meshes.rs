use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration, shader::*},
};
use kajiya_rg::{self as rg};
use rg::{Handle, RenderGraph};

use crate::{world_renderer::BlasInstance, hit_groups::new_rt_with_default_hit_groups};

use super::GbufferDepth;

pub struct RasterMeshesRtData<'a> {
    pub instances: &'a [BlasInstance],
    pub bindless_descriptor_set: vk::DescriptorSet,
}

pub fn rt_raster_meshes(
    rg: &mut RenderGraph,
    GbufferDepth {
        geometric_normal,
        gbuffer,
        depth,
        ..
    }: &mut GbufferDepth,
    velocity_img: &mut rg::Handle<Image>,
    tlas: &Handle<RayTracingAcceleration>,
    mesh_data: RasterMeshesRtData<'_>,
) {
    let instances = mesh_data.instances;

    new_rt_with_default_hit_groups(
        rg.add_pass("raster rt"),
        ShaderSource::hlsl("/shaders/rt/rt_raster.rgen.hlsl"),
        [
            ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
        ],
        true
    )
    .write(geometric_normal)
    .write(gbuffer)
    .write(depth)
    .write(velocity_img)
    .dynamic_storage_buffer_vec(
        instances
            .iter()
            .map(|inst| {
                let transform = [
                    inst.transformation.x_axis.x,
                    inst.transformation.y_axis.x,
                    inst.transformation.z_axis.x,
                    inst.transformation.translation.x,
                    inst.transformation.x_axis.y,
                    inst.transformation.y_axis.y,
                    inst.transformation.z_axis.y,
                    inst.transformation.translation.y,
                    inst.transformation.x_axis.z,
                    inst.transformation.y_axis.z,
                    inst.transformation.z_axis.z,
                    inst.transformation.translation.z,
                ];

                let prev_transform = [
                    inst.prev_transformation.x_axis.x,
                    inst.prev_transformation.y_axis.x,
                    inst.prev_transformation.z_axis.x,
                    inst.prev_transformation.translation.x,
                    inst.prev_transformation.x_axis.y,
                    inst.prev_transformation.y_axis.y,
                    inst.prev_transformation.z_axis.y,
                    inst.prev_transformation.translation.y,
                    inst.prev_transformation.x_axis.z,
                    inst.prev_transformation.y_axis.z,
                    inst.prev_transformation.z_axis.z,
                    inst.prev_transformation.translation.z,
                ];

                (transform, prev_transform)
            })
            .collect(),
    )
    .raw_descriptor_set(1, mesh_data.bindless_descriptor_set)
    .trace_rays(tlas, gbuffer.desc().extent);
}
