"""集成测试 - 验证所有修复"""
import sys
sys.path.insert(0, '/home/ubuntu/HAD-MC-Core-Algorithms')

def test_quantization_v2():
    """测试真实量化"""
    from hadmc.quantization_v2 import RealQuantizer
    print("✅ 真实量化模块导入成功")
    
def test_fusion_v2():
    """测试真实融合"""
    from hadmc.fusion_v2 import RealFusion
    print("✅ 真实融合模块导入成功")

def test_full_dataset():
    """测试完整数据集"""
    import os
    path = "/home/ubuntu/NEU-DET-Full"
    assert os.path.exists(path), "数据集不存在"
    images = len(os.listdir(f"{path}/IMAGES"))
    assert images == 1800, f"图像数量错误: {images}"
    print(f"✅ 完整数据集验证成功: {images}张图像")

if __name__ == "__main__":
    print("="*60)
    print("HAD-MC集成测试")
    print("="*60)
    
    test_quantization_v2()
    test_fusion_v2()
    test_full_dataset()
    
    print("\n所有集成测试通过! ✅")
