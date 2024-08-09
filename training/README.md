# 🤖 How to Train This Model Using Axolotl

<img src="https://github.com/user-attachments/assets/a8d0c95f-6268-42b9-93a4-ffea3c4e5987" alt="axolotl" width="140">

You need to set up the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework to start the training.

Follow the steps taken from [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) repostiry here:

```bash
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

For more details about set up: [Axolotl - Quickstart](https://github.com/axolotl-ai-cloud/axolotl?search=1#quickstart-)

# ⚙️ Use Our Config

Head to the [`nexus-qwen2-72b-instruct-orca.yaml`](https://github.com/NexusAI-tddi/NexusAI-tddi/blob/main/train/nexus-qwen2-72b-instruct-orca.yaml) file and download our YAML config.

Adjust the hyperparameters and other variables based on your setup and needs!

# 🖥️ Train the Model

You can start the training with the following code:
```bash
accelerate launch -m axolotl.cli.train nexus-qwen2-72b-instruct-orca.yaml
```
