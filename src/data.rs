use burn:{
    backend::{wgpu:AutoGraphicsApi, Autodiff, Wgpu},
    data:dataset::Dataset,
    optim:AdamConfig,
};
use guide::{model::ModelConfig, training::TrainingConfig};

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    guide::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    guide::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn:data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}


use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}