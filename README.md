### Installation and Setup

1. **Install Dependencies**
   To install the required Python dependencies, run:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Train the LoRA Model**
   To fine-tune the model, execute:
   ```bash
   python3 lora_train.py
   ```

3. **Start the vLLM Server**
   Use the following command to serve a model with LoRA enabled:
   ```bash
   vllm serve deepseek-ai/deepseek-llm-7b-chat \
     --enable-lora \
     --lora-modules redmine-lora=/path/to/lora_adapter \
     --uvicorn-log-level debug
   ```

4. **Deploy the Frontend Application**
   Run the following Docker command to start a Next.js-based UI:
   ```bash
   docker run --rm -d -p 3000:3000 \
     -e VLLM_URL=http://host.docker.internal:8000 \
     ghcr.io/yoziru/nextjs-vllm-ui:latest
   ```

---

### API Usage Example

To interact with the vLLM server, send a POST request using the following format:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
          "model": "/path/to/merged_model",
          "messages": [{"role": "user", "content": "List all my projects"}]
        }'
```