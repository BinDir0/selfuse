# LegendVLA-Inference

## Environment Setup

1. **Prerequisites**: Install Docker and download the image file.
2. **Load Image**: 
```bash
docker load -i teleop.tar
```
3. **Create Container**: Update the host path in `create_container.sh` (line 137) to your local project directory, then execute:
```bash
./create_container.sh legendvla-inference
```