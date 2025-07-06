# Garbage Classification Project (Garfinder)

Bu proje, farklı çöp türlerini sınıflandırmak için **MobileNetV2** mimarisiyle transfer öğrenmeyi kullanır. TensorFlow DirectML desteğiyle GPU hızlandırma sağlar.

## Özellikler

- MobileNetV2 tabanlı transfer learning  
- GPU destekli eğitim (DirectML)  
- %85+ test doğruluğu  
- Grafiklerle eğitim raporu

Modeli eğitmek için: egitim1.py veya  egitim2.py

Kaydedilen modeli test etmek için:  test.py

## Kullanım

### Kurulum

Önce ortamını hazırla:

```bash
pip install -r requirements.txt
