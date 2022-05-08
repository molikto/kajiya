use std::{
    collections::{hash_map, HashMap},
    sync::Arc,
};

use anyhow::Context;

use kajiya_backend::{
    vk_sync::{self, AccessType},
    Device, Image, ImageDesc,
};

use crate::ImportExportToRenderGraph;

use super::{
    Buffer, BufferDesc, ExportedHandle, Handle, RenderGraph, Resource, ResourceDesc,
    RetiredRenderGraph, TypeEquals,
};

pub struct ReadOnlyHandle<ResType: Resource>(Handle<ResType>);

impl<ResType: Resource> std::ops::Deref for ReadOnlyHandle<ResType> {
    type Target = Handle<ResType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<ResType: Resource> From<Handle<ResType>> for ReadOnlyHandle<ResType> {
    fn from(h: Handle<ResType>) -> Self {
        Self(h)
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct TemporalResourceKey(String);

impl<'a> From<&'a str> for TemporalResourceKey {
    fn from(s: &'a str) -> Self {
        TemporalResourceKey(String::from(s))
    }
}

impl<'a> From<String> for TemporalResourceKey {
    fn from(s: String) -> Self {
        TemporalResourceKey(s)
    }
}

pub(crate) enum TemporalResourceState<Res: Resource> {
    Inert {
        resource: Arc<Res>,
        access_type: AccessType,
    },
    Imported {
        resource: Arc<Res>,
        handle: Handle<Res>,
    },
    Exported {
        resource: Arc<Res>,
        handle: ExportedHandle<Res>,
    },
}

type ResMap<Res> = HashMap<TemporalResourceKey, TemporalResourceState<Res>>;

#[derive(Default)]
pub struct TemporalRenderGraphState {
    images: ResMap<Image>,
    buffers: ResMap<Buffer>,
}

impl TemporalRenderGraphState {
    fn clone_assuming_inert_resources<Res: Resource>(resources: &ResMap<Res>) -> ResMap<Res> {
        resources
            .iter()
            .map(|(k, v)| match v {
                TemporalResourceState::Inert {
                    resource,
                    access_type,
                } => (
                    k.clone(),
                    TemporalResourceState::Inert {
                        resource: resource.clone(),
                        access_type: *access_type,
                    },
                ),
                TemporalResourceState::Imported { .. } | TemporalResourceState::Exported { .. } => {
                    panic!("Not in inert state!")
                }
            })
            .collect()
    }
    pub(crate) fn clone_assuming_inert(&self) -> Self {
        Self {
            images: Self::clone_assuming_inert_resources(&self.images),
            buffers: Self::clone_assuming_inert_resources(&self.buffers),
        }
    }

    fn reuse_from_resources<Res: Resource>(self_map: &mut ResMap<Res>, other: ResMap<Res>) {
        for (res_key, res) in other {
            // `insert` is infrequent here, and we can avoid cloning the key.
            #[allow(clippy::map_entry)]
            if !self_map.contains_key(&res_key) {
                let res = match res {
                    res @ TemporalResourceState::Inert { .. } => res,
                    TemporalResourceState::Imported { resource, .. }
                    | TemporalResourceState::Exported { resource, .. } => {
                        TemporalResourceState::Inert {
                            resource,
                            access_type: vk_sync::AccessType::Nothing,
                        }
                    }
                };

                self_map.insert(res_key, res);
            }
        }
    }

    pub(crate) fn reuse_from(&mut self, temporal_rg_state: TemporalRenderGraphState) {
        let TemporalRenderGraphState { images, buffers } = temporal_rg_state;
        Self::reuse_from_resources(&mut self.buffers, buffers);
        Self::reuse_from_resources(&mut self.images, images);
    }
}

pub struct ExportedTemporalRenderGraphState(pub(crate) TemporalRenderGraphState);

pub struct TemporalRenderGraph {
    rg: RenderGraph,
    device: Arc<Device>,
    temporal_state: TemporalRenderGraphState,
}

impl std::ops::Deref for TemporalRenderGraph {
    type Target = RenderGraph;

    fn deref(&self) -> &Self::Target {
        &self.rg
    }
}

impl std::ops::DerefMut for TemporalRenderGraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.rg
    }
}

impl TemporalRenderGraph {
    pub fn new(state: TemporalRenderGraphState, device: Arc<Device>) -> Self {
        Self {
            rg: RenderGraph::new(),
            device,
            temporal_state: state,
        }
    }

    pub fn device(&self) -> &Device {
        self.device.as_ref()
    }
}

pub trait GetOrCreateTemporal<Desc: ResourceDesc> {
    fn get_or_create_temporal(
        &mut self,
        key: impl Into<TemporalResourceKey>,
        desc: Desc,
        //) -> anyhow::Result<Handle<Image>> {
    ) -> anyhow::Result<Handle<<Desc as ResourceDesc>::Resource>>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>;
}

impl GetOrCreateTemporal<ImageDesc> for TemporalRenderGraph {
    fn get_or_create_temporal(
        &mut self,
        key: impl Into<TemporalResourceKey>,
        desc: ImageDesc,
        //) -> anyhow::Result<Handle<Image>> {
    ) -> anyhow::Result<Handle<Image>> {
        let key = key.into();

        match self.temporal_state.images.entry(key.clone()) {
            hash_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();

                match state {
                    TemporalResourceState::Inert {
                        resource,
                        access_type,
                    } => {
                        let resource = resource.clone();
                        let handle = self.rg.import(resource.clone(), *access_type);
                        *state = TemporalResourceState::Imported {
                            resource,
                            handle: handle.clone_unchecked(),
                        };

                        Ok(handle)
                    }
                    TemporalResourceState::Imported { .. } => Err(anyhow::anyhow!(
                        "Temporal resource already taken: {:?}",
                        key
                    )),
                    TemporalResourceState::Exported { .. } => {
                        unreachable!()
                    }
                }
            }
            hash_map::Entry::Vacant(entry) => {
                let resource = Arc::new(
                    self.device
                        .create_image(desc, vec![])
                        .with_context(|| format!("Creating image {:?}", desc))?,
                );
                let handle = self.rg.import(resource.clone(), AccessType::Nothing);
                entry.insert(TemporalResourceState::Imported {
                    resource,
                    handle: handle.clone_unchecked(),
                });
                Ok(handle)
            }
        }
    }
}

impl GetOrCreateTemporal<BufferDesc> for TemporalRenderGraph {
    fn get_or_create_temporal(
        &mut self,
        key: impl Into<TemporalResourceKey>,
        desc: BufferDesc,
        //) -> anyhow::Result<Handle<Image>> {
    ) -> anyhow::Result<Handle<Buffer>> {
        let key = key.into();

        match self.temporal_state.buffers.entry(key.clone()) {
            hash_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();

                match state {
                    TemporalResourceState::Inert {
                        resource,
                        access_type,
                    } => {
                        let resource = resource.clone();
                        let handle = self.rg.import(resource.clone(), *access_type);

                        *state = TemporalResourceState::Imported {
                            resource,
                            handle: handle.clone_unchecked(),
                        };

                        Ok(handle)
                    }
                    TemporalResourceState::Imported { .. } => Err(anyhow::anyhow!(
                        "Temporal resource already taken: {:?}",
                        key
                    )),
                    TemporalResourceState::Exported { .. } => {
                        unreachable!()
                    }
                }
            }
            hash_map::Entry::Vacant(entry) => {
                let resource = Arc::new(self.device.create_buffer(desc, &key.0, None)?);
                let handle = self.rg.import(resource.clone(), AccessType::Nothing);
                entry.insert(TemporalResourceState::Imported {
                    resource: resource,
                    handle: handle.clone_unchecked(),
                });
                Ok(handle)
            }
        }
    }
}

impl TemporalRenderGraph {
    fn export_temporal_resources<Res: ImportExportToRenderGraph>(
        rg: &mut RenderGraph,
        resources: &mut ResMap<Res>,
    ) {
        for state in resources.values_mut() {
            match state {
                TemporalResourceState::Inert { .. } => {
                    // Nothing to do here
                }
                TemporalResourceState::Imported { resource, handle } => {
                    let handle = rg.export(handle.clone_unchecked(), AccessType::Nothing);
                    *state = TemporalResourceState::Exported {
                        resource: resource.clone(),
                        handle,
                    }
                }
                TemporalResourceState::Exported { .. } => {
                    unreachable!()
                }
            }
        }
    }
    pub fn export_temporal(self) -> (RenderGraph, ExportedTemporalRenderGraphState) {
        let mut rg = self.rg;
        let mut state = self.temporal_state;
        Self::export_temporal_resources(&mut rg, &mut state.images);
        Self::export_temporal_resources(&mut rg, &mut state.buffers);
        (rg, ExportedTemporalRenderGraphState(state))
    }
}

impl ExportedTemporalRenderGraphState {
    fn retire_temporal_resources<Res: ImportExportToRenderGraph + Resource>(
        rg: &RetiredRenderGraph,
        resources: &mut ResMap<Res>,
    ) {
        for state in resources.values_mut() {
            match state {
                TemporalResourceState::Inert { .. } => {
                    // Nothing to do here
                }
                TemporalResourceState::Imported { .. } => {
                    unreachable!()
                }
                TemporalResourceState::Exported { resource, handle } => {
                    *state = TemporalResourceState::Inert {
                        resource: resource.clone(),
                        access_type: rg.exported_resource(*handle).1,
                    }
                }
            }
        }
    }
    pub fn retire_temporal(self, rg: &RetiredRenderGraph) -> TemporalRenderGraphState {
        let mut state = self.0;
        Self::retire_temporal_resources(rg, &mut state.images);
        Self::retire_temporal_resources(rg, &mut state.buffers);
        state
    }
}
