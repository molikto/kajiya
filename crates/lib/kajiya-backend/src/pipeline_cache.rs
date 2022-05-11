use crate::{
    rust_shader_compiler::CompileRustShader,
    shader_compiler::{CompileShader, CompiledShader},
    vulkan::{
        ray_tracing::{create_ray_tracing_pipeline, RayTracingPipeline, RayTracingPipelineDesc},
        shader::*,
    },
};
use bytes::Bytes;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{collections::HashMap, sync::Arc};
use turbosloth::*;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ComputePipelineHandle(usize);

struct ComputePipelineCacheEntry {
    lazy_handle: Lazy<CompiledShader>,
    desc: ComputePipelineDesc,
    pipeline: Option<Arc<ComputePipeline>>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RasterPipelineHandle(usize);

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RtPipelineHandle(usize);

async fn compile(
    desc: PipelineShaderDesc,
    profile: &'static str,
    ctx: &RunContext,
) -> anyhow::Result<PipelineShader<Bytes>> {
    let compiled: Arc<CompiledShader> = match &desc.source {
        ShaderSource::Rust { entry } => CompileRustShader {
            entry: entry.clone(),
        }
        .into_lazy()
        .eval(ctx),
        ShaderSource::Hlsl { path } => CompileShader {
            path: path.clone(),
            profile: profile.to_string(),
        }
        .into_lazy()
        .eval(ctx),
    }
    .await?;
    Ok(PipelineShader {
        desc,
        code: compiled.spirv.clone(),
    })
}
#[async_trait]
impl LazyWorker for RasterPipelineShadersDesc {
    type Output = anyhow::Result<RasterPipelineShaders<Bytes>>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let vertex = compile(self.vertex, "vs", &ctx).await?;
        let pixel = compile(self.pixel, "ps", &ctx).await?;
        Ok(RasterPipelineShaders { vertex, pixel })
    }
}

#[async_trait]
impl LazyWorker for RayTracingPipelineShadersDesc {
    type Output = anyhow::Result<Vec<PipelineShaderGroup<Bytes>>>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let mut groups = Vec::new();
        for g in self.0 {
            match g {
                ShaderGroupDesc::RayGen(g) => {
                    groups.push(PipelineShaderGroup::RayGen(compile(g, "lib", &ctx).await?));
                }
                ShaderGroupDesc::Miss(g) => {
                    groups.push(PipelineShaderGroup::Miss(compile(g, "lib", &ctx).await?));
                }
                ShaderGroupDesc::HitGroup {
                    closest_hit,
                    intersection,
                } => groups.push(PipelineShaderGroup::HitGroup {
                    cloest_hit: match closest_hit {
                        Some(g) => Some(compile(g, "lib", &ctx).await?),
                        None => None,
                    },
                    intersection: match intersection {
                        Some(g) => Some(compile(g, "lib", &ctx).await?),
                        None => None,
                    },
                }),
            }
        }
        Ok(groups)
    }
}

struct RasterPipelineCacheEntry {
    lazy_handle: Lazy<RasterPipelineShaders<Bytes>>,
    desc: RasterPipelineDesc,
    pipeline: Option<Arc<RasterPipeline>>,
}

struct RtPipelineCacheEntry {
    lazy_handle: Lazy<Vec<PipelineShaderGroup<Bytes>>>,
    desc: RayTracingPipelineDesc,
    pipeline: Option<Arc<RayTracingPipeline>>,
}

pub struct PipelineCache {
    lazy_cache: Arc<LazyCache>,

    compute_entries: HashMap<ComputePipelineHandle, ComputePipelineCacheEntry>,
    raster_entries: HashMap<RasterPipelineHandle, RasterPipelineCacheEntry>,
    rt_entries: HashMap<RtPipelineHandle, RtPipelineCacheEntry>,

    compute_shader_to_handle: HashMap<ShaderSource, ComputePipelineHandle>,
    raster_shaders_to_handle: HashMap<RasterPipelineShadersDesc, RasterPipelineHandle>,
    rt_shaders_to_handle: HashMap<Vec<ShaderGroupDesc>, RtPipelineHandle>,
}

impl PipelineCache {
    pub fn new(lazy_cache: &Arc<LazyCache>) -> Self {
        Self {
            lazy_cache: lazy_cache.clone(),

            compute_entries: Default::default(),
            raster_entries: Default::default(),
            rt_entries: Default::default(),

            compute_shader_to_handle: Default::default(),

            raster_shaders_to_handle: Default::default(),
            rt_shaders_to_handle: Default::default(),
        }
    }

    // TODO: should probably use the `desc` as key as well
    pub fn register_compute(&mut self, desc: &ComputePipelineDesc) -> ComputePipelineHandle {
        match self.compute_shader_to_handle.entry(desc.source.clone()) {
            std::collections::hash_map::Entry::Occupied(occupied) => *occupied.get(),
            std::collections::hash_map::Entry::Vacant(vacant) => {
                let handle = ComputePipelineHandle(self.compute_entries.len());
                let compile_task = match &desc.source {
                    ShaderSource::Rust { entry } => CompileRustShader {
                        entry: entry.clone(),
                    }
                    .into_lazy(),
                    ShaderSource::Hlsl { path } => CompileShader {
                        path: path.clone(),
                        profile: "cs".to_owned(),
                    }
                    .into_lazy(),
                };

                self.compute_entries.insert(
                    handle,
                    ComputePipelineCacheEntry {
                        lazy_handle: compile_task,
                        desc: desc.clone(),
                        pipeline: None,
                    },
                );
                vacant.insert(handle);
                handle
            }
        }
    }

    pub fn get_compute(&self, handle: ComputePipelineHandle) -> Arc<ComputePipeline> {
        self.compute_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    pub fn register_raster(
        &mut self,
        shaders: &RasterPipelineShadersDesc,
        desc: &RasterPipelineDesc,
    ) -> RasterPipelineHandle {
        if let Some(handle) = self.raster_shaders_to_handle.get(shaders) {
            return *handle;
        }

        let handle = RasterPipelineHandle(self.raster_entries.len());
        self.raster_shaders_to_handle
            .insert(shaders.to_owned(), handle);
        self.raster_entries.insert(
            handle,
            RasterPipelineCacheEntry {
                lazy_handle: shaders.clone().into_lazy(),
                desc: desc.clone(),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get_raster(&self, handle: RasterPipelineHandle) -> Arc<RasterPipeline> {
        self.raster_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    pub fn register_ray_tracing(
        &mut self,
        shaders: &Vec<ShaderGroupDesc>,
        desc: &RayTracingPipelineDesc,
    ) -> RtPipelineHandle {
        if let Some(handle) = self.rt_shaders_to_handle.get(shaders) {
            return *handle;
        }

        let handle = RtPipelineHandle(self.rt_entries.len());
        self.rt_shaders_to_handle.insert(shaders.to_owned(), handle);
        self.rt_entries.insert(
            handle,
            RtPipelineCacheEntry {
                lazy_handle: RayTracingPipelineShadersDesc(shaders.to_vec()).into_lazy(),
                desc: desc.clone(),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get_ray_tracing(&self, handle: RtPipelineHandle) -> Arc<RayTracingPipeline> {
        self.rt_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    fn invalidate_stale_pipelines(&mut self) {
        for entry in self.compute_entries.values_mut() {
            if entry.pipeline.is_some() && entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }
        }

        for entry in self.raster_entries.values_mut() {
            if entry.pipeline.is_some() && entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }
        }

        for entry in self.rt_entries.values_mut() {
            if entry.pipeline.is_some() && entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }
        }
    }

    pub fn parallel_compile_shaders(
        &mut self,
        device: &Arc<crate::vulkan::device::Device>,
    ) -> anyhow::Result<()> {
        // Prepare build tasks for compute
        let compute = self.compute_entries.iter().filter_map(|(&handle, entry)| {
            entry.pipeline.is_none().then(|| {
                let task = entry.lazy_handle.eval(&self.lazy_cache);
                smol::spawn(async move {
                    task.await.map(|compiled| CompileTaskOutput::Compute {
                        handle,
                        compiled: compiled.spirv.clone(),
                    })
                })
            })
        });

        // Prepare build tasks for raster
        let raster = self.raster_entries.iter().filter_map(|(&handle, entry)| {
            entry.pipeline.is_none().then(|| {
                let task = entry.lazy_handle.eval(&self.lazy_cache);
                smol::spawn(async move {
                    task.await
                        .map(|compiled| CompileTaskOutput::Raster { handle, compiled })
                })
            })
        });

        // Prepare build tasks for rt
        let rt = self.rt_entries.iter().filter_map(|(&handle, entry)| {
            entry.pipeline.is_none().then(|| {
                let task = entry.lazy_handle.eval(&self.lazy_cache);
                smol::spawn(async move {
                    task.await
                        .map(|compiled| CompileTaskOutput::Rt { handle, compiled })
                })
            })
        });

        // Gather all the build tasks together
        let shader_tasks: Vec<_> = compute.chain(raster).chain(rt).collect();

        if !shader_tasks.is_empty() {
            // Compile all the things
            let compiled: Vec<CompileTaskOutput> =
                smol::block_on(futures::future::try_join_all(shader_tasks))?;

            // Build pipelines from all compiled shaders
            for compiled in compiled {
                match compiled {
                    CompileTaskOutput::Compute { handle, compiled } => {
                        let entry = self.compute_entries.get_mut(&handle).unwrap();
                        log::trace!(
                            "Creating compute pipeline {:?}",
                            entry.desc.source.entry(),
                        );
                        entry.pipeline = Some(Arc::new(create_compute_pipeline(
                            &*device,
                            &compiled,
                            &entry.desc,
                        )));
                    }
                    CompileTaskOutput::Raster { handle, compiled } => {
                        let entry = self.raster_entries.get_mut(&handle).unwrap();
                        log::trace!(
                            "Creating raster pipeline",
                        );
                        // TODO: defer and handle the error
                        entry.pipeline = Some(Arc::new(
                            create_raster_pipeline(&*device, &compiled, &entry.desc)
                                .expect("create_raster_pipeline"),
                        ));
                    }
                    CompileTaskOutput::Rt { handle, compiled } => {
                        let entry = self.rt_entries.get_mut(&handle).unwrap();
                        log::trace!(
                            "Creating rt pipeline",
                        );
                        // TODO: defer and handle the error
                        entry.pipeline = Some(Arc::new(
                            create_ray_tracing_pipeline(&*device, &compiled, &entry.desc)
                                .expect("create_ray_tracing_pipeline"),
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn prepare_frame(
        &mut self,
        device: &Arc<crate::vulkan::device::Device>,
    ) -> anyhow::Result<()> {
        self.invalidate_stale_pipelines();
        self.parallel_compile_shaders(device)?;

        Ok(())
    }
}

enum CompileTaskOutput {
    Compute {
        handle: ComputePipelineHandle,
        compiled: Bytes,
    },
    Raster {
        handle: RasterPipelineHandle,
        compiled: Arc<RasterPipelineShaders<Bytes>>,
    },
    Rt {
        handle: RtPipelineHandle,
        compiled: Arc<Vec<PipelineShaderGroup<Bytes>>>,
    },
}
