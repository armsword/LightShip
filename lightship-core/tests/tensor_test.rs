//! Unit tests for Tensor

use lightship_core::common::{DataType, StorageLayout, TensorLifetime};
use lightship_core::ir::Tensor;

#[test]
fn test_tensor_creation() {
    let tensor = Tensor::new("test".into(), vec![1, 3, 224, 224], DataType::F32);

    assert_eq!(tensor.name, "test");
    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
    assert_eq!(tensor.data_type, DataType::F32);
    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.num_elements(), 1 * 3 * 224 * 224);
}

#[test]
fn test_tensor_byte_size() {
    let tensor = Tensor::new("test".into(), vec![2, 3, 32, 32], DataType::F32);

    // 2*3*32*32 = 6144 elements * 4 bytes = 24576 bytes
    assert_eq!(tensor.byte_size(), 2 * 3 * 32 * 32 * 4);
}

#[test]
fn test_tensor_data_type_helpers() {
    let f32_tensor = Tensor::new("f32".into(), vec![1], DataType::F32);
    let i8_tensor = Tensor::new("i8".into(), vec![1], DataType::I8);
    let qint8_tensor = Tensor::new("qint8".into(), vec![1], DataType::QInt8);

    assert!(f32_tensor.data_type.is_float());
    assert!(!f32_tensor.data_type.is_quantized());

    assert!(i8_tensor.data_type.is_int());
    assert!(!i8_tensor.data_type.is_quantized());

    assert!(qint8_tensor.data_type.is_quantized());
}

#[test]
fn test_tensor_static_lifetime() {
    let mut tensor = Tensor::new("weight".into(), vec![64, 3, 7, 7], DataType::F32);
    assert!(!tensor.is_static());

    tensor.lifetime = TensorLifetime::Static;
    assert!(tensor.is_static());
}
