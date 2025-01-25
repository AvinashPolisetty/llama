from typing import Optional
from pathlib import Path
from tqdm import tqdm
import time
import torch
import json,os,sys
from sentencepiece import SentencePieceProcessor
from model import ModelArgs,Transformer

class LLama:

    def __init__(self,model:Transformer,tokenizer:SentencePieceProcessor,model_args:ModelArgs)
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(chkpt_dir:str,tokenizer_pt:str,load_model:bool,max_seq_len:int,max_batch_size:int,device:str):
        prev_time = time.time()
        if load_model:
            checkpts = sorted(Path(chkpt_dir).glob('*.pth'))
            assert len(checkpts) > 0 ,"no checkpoints files are found"
            chk_path = checkpts[0]
            print(f'Loading Checkpoint {chk_path}')
            checkpoint = torch.load(chk_path,map_location="cpu")
            print(f"Checkpoints loaded in {(time.time() - prev_time): .3f}s")
            prev_time = time.time()
        
        with open(Path(chkpt_dir)/"params.json",'r') as f:
            params = json.load(f.read())
        model_args:ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            ** params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_pt)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint,strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time): .3f}s")

        return LLama(model,tokenizer,model_args)
    

if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    model = LLama.build(
        chkpt_dir="sa",
        tokenizer_pt="",
        load_model=True,
        max_seq_len=10,
        max_batch_size=2,
        device="cpu"    
        )
