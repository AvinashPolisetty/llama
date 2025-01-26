from typing import Optional,List
from pathlib import Path
from tqdm import tqdm
import time
import torch
import json,os,sys
from sentencepiece import SentencePieceProcessor
from model import ModelArgs,Transformer

class LLama:

    def __init__(self,model:Transformer,tokenizer:SentencePieceProcessor,model_args:ModelArgs):
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


    def text_completion(self,prompts:list[str],temp:float = 0.6,top_p:float = 0.9,max_gen_len:Optional[int]=None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len-1


            prompt_tokens = [self.tokenizer.encode(prompt,out_type=int,add_bos=True,add_eos = False) for prompt in prompts]

            batch_size = len(prompt_tokens)
            assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
            max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
            # Make sure the prompt length is not larger than the maximum sequence length
            assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
            total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

            # Create the list that will contain the generated tokens, along with the initial prompt tokens
            pad_id = self.tokenizer.pad_id()
            tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
            for k, t in enumerate(prompt_tokens):
                # Populate the initial tokens with the prompt tokens
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            
            eos_reached = torch.tensor([False] * batch_size, device=device)
            prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
            cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
            for cur_pos in cur_iterator:
                with torch.no_grad():
                    logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
                if temp > 0:
                    # The temperature is applied before the softmax
                    probs = torch.softmax(logits[:, -1] / temp, dim=-1)
                    next_token = self._sample_top_p(probs, top_p)
                else:
                    # Greedily select the token with the max probability
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                # Only replace token if it is a padding token
                next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token
                # EOS is reached only if we found an EOS token for a padding position
                eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
                if all(eos_reached):
                    break

            out_tokens = []
            out_text = []
            for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
                # Cut to the EOS token, if present
                if self.tokenizer.eos_id in current_prompt_tokens:
                    eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                    current_prompt_tokens = current_prompt_tokens[:eos_idx]
                out_tokens.append(current_prompt_tokens)
                out_text.append(self.tokenizer.decode(current_prompt_tokens))
            return (out_tokens, out_text)
        

    def sample_top_p(self,probs,p):

        probs_sort,probs_idx = torch.sort(probs,dim=-1,descending=True)
        probs_sum = torch.cumsum(probs_sort,dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))
        next_token = torch.multinomial(probs_sort,num_samples=1)
        next_token = torch.gather(probs_idx,-1,next_token)
        return next_token



if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to german:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Avinash
        Decision: 
        """
    ]

    model = LLama.build(
        chkpt_dir="sa",
        tokenizer_pt="",
        load_model=True,
        max_seq_len=10,
        max_batch_size=2,
        device="cpu"    
        )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
