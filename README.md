# NexusAI

<img src="https://github.com/user-attachments/assets/5be90254-14e6-44b6-94a2-19460fd8433f" alt="NexusAI" width="400" height="400"> 

# 📂 Proje Tanımı:

Günümüzde yapay zeka, farklı dillerdeki metinleri anlama ve işleme konusunda büyük ilerlemeler kaydetmiştir. Ancak, açık kaynaklı base modellerin **çok az Türkçe veriyle eğitilmesi** sonucunda modeller Türkçe anlama ve konuşmada sıkıntılar yaşamaktadır.

Bunun sonucunda __NexusAI__, **Türkçe anlama ve konuşması gelişmis** bir model eğitmiştir.

Bu modelle birlikte, diğer yapay zeka modelleriyle hızlı bir şekilde çalıştırıldığında, karşımıza **hem Türkçe anlayıp konuşabilen** hem de **resimleri anlayabilen** bir ürün ortaya çıkmış olacaktır.

# 💡 Projenin Kullanım Alanları ve Sağladığı Çözümler

- Büyük dil modelleri, çeşitli modeller sayesinde resim yorumlayabilir hale gelir.

- Dil modelinin güçlü yazı performansı etkilenmez, görsel analiz yapılabilir.

- Kurumsal yazılımlar, sağlık ve güvenlik sektörlerinde rahatlıkla kullanılabilir.

Görsel analiz yapabilen modeller, kişiselleştirme ve müşteri memnuniyetini üst seviyelere çıkararak daha etkili ve verimli çözümler sunar.

# 📊 Veri Seti

#### Türkçe modelin eğitimi için **OpenOrca** veri setinin __Türkçe’ye çevirilmiş__ versiyonu kullanılmıştır. 

Bu veri setine bu bağlantıyı kullanarak ulaşabilirsiniz: [NexusAI-tddi/OpenOrca-tr-1-million-sharegpt](https://huggingface.co/datasets/NexusAI-tddi/OpenOrca-tr-1-million-sharegpt)

#### Geliştirilen bütün sistemi test etmek için kullanılan veri seti

Bütün modelin test edilmesi için VisIT-Bench veri setinin Türkçe’ye çevirilmiş versiyonu kullanılmıştır.

Bu veri setine bu bağlantıyı kullanarak ulaşabilirsiniz: [NexusAI-tddi/VisIT-Bench-tr](https://huggingface.co/datasets/NexusAI-tddi/VisIT-Bench-tr)

Veri setini işlemek için kullanılan kodu [`dataset`](https://github.com/NexusAI-tddi/NexusAI-tddi/tree/main/dataset) klasörünün içinde bulabilirsiniz.

# 🔧 Yöntem ve Teknikler:

[QLoRa:](https://arxiv.org/abs/2305.14314) Önceden eğitilmiş LLM'deki ağırlık parametrelerinin hassasiyetini 4 bit hassasiyete indirgeyerek kullanmaya yarar. LoRA'nın özel bir hâlidir.

[Flash Attention:](https://arxiv.org/abs/2205.14135) Özellikle büyük dil modellerinin verimliliğini artırmak ve bellek gereksinimlerini azaltmak için tasarlanmış bir dikkat algoritmasıdır.

[Cosine Similarity:](https://www.sciencedirect.com/topics/computer-science/cosine-similarity) İki vektör arasındaki açıya dayalı benzerliği ölçen bir benzerlik metriğidir.,

## ⛏️ Kullanılan Modeller

### 👁️ Nesnelerin Algılanması:

- **YOLOv8**: Fotoğraftaki nesnelerin algılanması için YOLOv8 modeli kullanılmıştır. [YOLOv8 Github Repo](https://github.com/ultralytics/ultralytics)

### 📸 Nesnelerin ve Fotoğrafın Açıklamasının Oluşturulması:

- **Moondream2**: Fotoğrafın tamamı ve nesnelerin açıklamasının oluşturulması için kullanılmıştır. [Moondream2 Modeli](https://huggingface.co/vikhyatk/moondream2)

### 🤖 Büyük Dil Modeli

- **Qwen2-72B-Instruct**, base model olarak kullanıldı.

# 🤖 Model Eğitimi

### Modelin Eğitim Tekniği:
- [QLoRA (Efficient Finetuning of Quantized LLMs)](https://arxiv.org/abs/2305.14314)
- [8bitAdamW Optimizer](https://huggingface.co/docs/bitsandbytes/main/en/optimizers#8-bit-optimizers)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

### Eğitim Kütüphanesi:
- [Transformers](https://github.com/huggingface/transformers) 
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)

### Base Model

Önceden eğitilmiş [Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) modelini diğer modellere kıyasla, Türkçe için daha uygun bulduk.

Eğitim ile ilgili daha fazla bilgi ve konfigürasyonumuz için [`training`](https://github.com/NexusAI-tddi/NexusAI-tddi/tree/main/training) klasörüne bakabilirsiniz.

# 🔍 Değerlendirme

Aşağıdaki confusion matrix'ten gerekli değerlendirmeleri çıkarabilirsiniz.

![asil](https://github.com/user-attachments/assets/68f382b0-1c5c-4713-a848-f96b13dd7abf)

# 🏆 Sonuçlar

Eğitilen **Türkçe** büyük dil modeli ve işlenen veri seti, **Hugging Face organizasyon** hesabında paylaşılmıştır.

Proje, görsellerle ile ilgili çeşitli metriklerde ve gerçek hayat uygulamalarında **üstün performans** göstermiştir.

Bu projenin, hem **Türkçe dilinde** olması hem de **görsel anlayabilmesi** özellikleri sayesinde, **birçok sektörde aktif olarak** kullanılabileceği sonucuna varılmıştır.

- İşlenen veri seti: [`NexusAI-tddi/OpenOrca-tr-1-million-sharegpt`](https://huggingface.co/datasets/NexusAI-tddi/OpenOrca-tr-1-million-sharegpt)

- Eğitilen **Türkçe** Büyük Dil Modeli: [`NexusAI-tddi/Qwen2-72B-Instruct-OpenOrca-tr`](https://huggingface.co/NexusAI-tddi/Qwen2-72B-Instruct-OpenOrca-tr)

# 🎥 Demo Uygulaması

Sohbet tabanlı bu arayüzde, metin ve görsel girişler kolaylıkla yapılabilmektedir. 

- **Huggingface Spaces**: Demo uygulamamızı Huggingface Spaces'de barındırıp, geliştiriyoruz. [Huggingface Spaces Sayfası](https://huggingface.co/spaces)
- **Gradio Framework**: Arayüzümüz geliştirilirken, Python tabanlı Gradio framework'ü kullanılmıştır. [Gradio PyPI](https://pypi.org/project/gradio/)

Demo ile ilgili daha fazla bilgi için [`demo-app`](https://github.com/NexusAI-tddi/NexusAI-tddi/tree/main/demo-app) klasörüne bakabilirsiniz.

## Demo Videosu

https://github.com/user-attachments/assets/3dc484a7-2af1-4df6-a0b5-ff69f5a9166f
