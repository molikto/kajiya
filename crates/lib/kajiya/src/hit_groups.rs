use kajiya_backend::vulkan::shader::{HitGroupShaderSources, ShaderSource};
use kajiya_rg::*;

pub fn new_rt_with_default_hit_groups<'rg>(
    pass: PassBuilder<'rg>,
    rgen: ShaderSource,
    miss: impl IntoIterator<Item = ShaderSource>,
    has_hit_groups: bool,
) -> SimpleRenderPass<'rg, RgRtPipelineHandle> {
    SimpleRenderPass::new_rt(
        pass,
        rgen,
        miss,
        if has_hit_groups {
            vec![
                HitGroupShaderSources {
                    closest_hit: Some(ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")),
                    intersection: None,
                },
                HitGroupShaderSources {
                    closest_hit: Some(ShaderSource::hlsl("/shaders/default_aabb.rchit.hlsl")),
                    intersection: Some(ShaderSource::hlsl("/shaders/default_aabb.rint.hlsl")),
                },
            ]
        } else {
            vec![
                HitGroupShaderSources {
                    closest_hit: None,
                    intersection: None,
                },
                HitGroupShaderSources {
                    closest_hit: None,
                    intersection: Some(ShaderSource::hlsl("/shaders/default_aabb.rint.hlsl")),
                },
            ]
        },
    )
}
