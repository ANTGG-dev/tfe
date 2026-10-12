import gc
import time
import numpy as np
import torch
from ultralytics.utils.torch_utils import get_flops

def profile(yolo,imgsz=768,warmup=20,repeats=100):
    net=yolo.model.cuda().float().eval()
    params=sum(p.numel() for p in net.parameters());flops=get_flops(net,imgsz)
    # Ultralytics fuse includes supported RepConv reparameterization; TFE stays branched.
    net.fuse(verbose=False)
    x=torch.zeros(1,3,imgsz,imgsz,device='cuda');samples=[]
    with torch.inference_mode():
        for _ in range(warmup):net(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();net(x);end.record();torch.cuda.synchronize();samples.append(start.elapsed_time(end))
    peak=torch.cuda.max_memory_allocated()/1024**2
    # Separate end-to-end measurement includes preprocessing, transfer, decoder/NMS,
    # coordinate restoration; excludes image reading, rendering and serialization.
    image=np.zeros((640,640,3),dtype=np.uint8);pipeline=[]
    for i in range(warmup+repeats):
        torch.cuda.synchronize();start=time.perf_counter();yolo.predict(image,imgsz=imgsz,device=0,conf=.001,iou=.7,max_det=300,half=False,verbose=False);torch.cuda.synchronize()
        if i>=warmup:pipeline.append((time.perf_counter()-start)*1000)
    return {'params':params,'GFLOPs':flops,'input_size':imgsz,'batch':1,'precision':'FP32','backend':'PyTorch CUDA eager','warmup':warmup,'iterations':repeats,'fused_supported_layers':True,'TFE_structurally_merged':False,'forward_mean_ms':float(np.mean(samples)),'forward_p50_ms':float(np.median(samples)),'forward_p95_ms':float(np.quantile(samples,.95)),'forward_peak_allocated_mb':peak,'pipeline_mean_ms':float(np.mean(pipeline)),'pipeline_p95_ms':float(np.quantile(pipeline,.95)),'pipeline_includes':'preprocess + H2D + model + decoder/NMS when applicable + coordinate restoration','end_to_end_model':bool(getattr(net,'end2end',False)),'power_budget_note':'desktop RTX measurements do not establish UAV/Jetson power, latency or endurance'}
