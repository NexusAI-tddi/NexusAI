# 🤖 Bu Modeli Axolotl Kullanarak Nasıl Eğitebilirsiniz?

<img src="https://github.com/user-attachments/assets/a8d0c95f-6268-42b9-93a4-ffea3c4e5987" alt="axolotl" width="140">

Eğitime başlamak için [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework'ünü kurmanız gerekmektedir.

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) repo'sundan alınan adımları takip edin:

```bash
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

Kurulum hakkında daha fazla bilgi için: [Axolotl - Quickstart](https://github.com/axolotl-ai-cloud/axolotl?search=1#quickstart-)

# ⚙️ Yapılandırmamızı Kullanın

[nexus-qwen2-72b-instruct-orca.yaml](https://github.com/NexusAI-tddi/NexusAI-tddi/blob/main/train/nexus-qwen2-72b-instruct-orca.yaml) dosyasına gidin ve YAML dosyamızı indirin.

Kendi kurulumunuza ve ihtiyaçlarınıza göre hiperparametreleri ve diğer değişkenleri ayarlayın!

# 🖥️ Modeli Eğitin

Eğitime aşağıdaki kodla başlayabilirsiniz:

```bash
accelerate launch -m axolotl.cli.train nexus-qwen2-72b-instruct-orca.yaml
```
