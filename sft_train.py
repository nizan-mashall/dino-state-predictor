import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import sys
from peft import LoraConfig, get_peft_model
from prismatic.vla.action_tokenizer import ActionTokenizer
import time


os.makedirs('/code/checkpoints', exist_ok=True)

class RoboSuiteDataset(Dataset):
    def __init__(self, hdf5_path, processor, instruction):
        self.f = h5py.File(hdf5_path, 'r')
        self.demos = list(self.f['data'].keys())[:20]
        self.instruction = instruction
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.index = []
        for demo_key in self.demos:
            T = len(self.f['data'][demo_key]['actions'])
            for t in range(T):
                self.index.append((demo_key, t))
        print(f"Dataset: {len(self.demos)} demos, {len(self.index)} total steps")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        demo_key, t = self.index[idx]
        #image = self.f['data'][demo_key]['obs']['robot0_eye_in_hand_image'][t]
        image = self.f['data'][demo_key]['obs']['agentview_image'][t]
        action = self.f['data'][demo_key]['actions'][t]
        return {
            'image': Image.fromarray(image.astype(np.uint8)),
            'action': action,
            'instruction': self.instruction
        }

def collate_fn(batch, processor, action_tokenizer):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_pixel_values = []
    
    DEBUG = False  # set to False to disable prints
    
    for item in batch:
        image = item['image']
        action = item['action']
        instruction = item['instruction']
        
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        batch_inputs = processor(text=prompt, images=image, return_tensors='pt')
        
        input_ids = batch_inputs['input_ids'][0]
        attention_mask = batch_inputs['attention_mask'][0]
        pixel_values = batch_inputs['pixel_values'][0]
        
        # Tokenize action directly
        Q01 = np.array([-0.2737, -0.2854, -0.8444, -0.0562, -0.0814, -0.1755, -1.0])
        Q99 = np.array([ 0.7964,  0.3604,  1.0,     0.0558,  0.1694,  0.2551,  1.0])

        action_normalized = 2 * (action - Q01) / (Q99 - Q01 + 1e-8) - 1
        action_normalized = np.clip(action_normalized, -1, 1)
        discretized = np.digitize(action_normalized, action_tokenizer.bins)
        action_token_ids = processor.tokenizer.vocab_size - discretized
        
        if DEBUG:
            print(f"\n--- DEBUG collate_fn ---")
            print(f"Original action:    {action.round(4)}")
            print(f"Discretized bins:   {discretized}")
            print(f"Action token IDs:   {action_token_ids.tolist()}")
            print(f"Num action tokens:  {len(action_token_ids)}")
            print(f"All action tokens?: {all(t > action_tokenizer.action_token_begin_idx for t in action_token_ids)}")
            print(f"Prompt token IDs:   {input_ids.tolist()}")
            print(f"Prompt length:      {len(input_ids)}")
            
        action_tensor = torch.tensor(action_token_ids, dtype=torch.long)
        full_input_ids = torch.cat([input_ids, action_tensor], dim=0)
        full_attention_mask = torch.cat([attention_mask, torch.ones(len(action_token_ids), dtype=torch.long)], dim=0)
        
        labels = full_input_ids.clone()
        #labels[:-(len(action_token_ids) + 1)] = -100
        labels[:-(len(action_token_ids))] = -100

        if DEBUG:
            print(f"Full sequence length: {len(full_input_ids)}")
            print(f"Active label tokens:  {(labels != -100).sum().item()}")
            print(f"Active label IDs:     {full_input_ids[labels != -100].tolist()}")
            print(f"Decoded active:       {processor.tokenizer.decode(full_input_ids[labels != -100].tolist())}")
            DEBUG = False  # only print for first item in batch
        
        all_input_ids.append(full_input_ids)
        all_attention_masks.append(full_attention_mask)
        all_labels.append(labels)
        all_pixel_values.append(pixel_values)
    
    # 5. Right Padding
    max_len = max(ids.shape[0] for ids in all_input_ids)
    
    padded_input_ids = torch.full((len(batch), max_len), processor.tokenizer.pad_token_id, dtype=torch.long)
    padded_attention_masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i in range(len(batch)):
        seq_len = all_input_ids[i].shape[0]
        padded_input_ids[i, :seq_len] = all_input_ids[i]
        padded_attention_masks[i, :seq_len] = all_attention_masks[i]
        padded_labels[i, :seq_len] = all_labels[i]
    
    # This dictionary payload was missing, causing the NoneType crash
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'pixel_values': torch.stack(all_pixel_values),
        'labels': padded_labels
    }


def train():

    device = torch.device('cuda')
    print("Loading processor and model...")
    
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.0,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)
    
    dataset = RoboSuiteDataset(
        '/code/data/demo_with_images.hdf5',
        processor,              # add processor
        'pick up the red cube'
    )

    dataloader = DataLoader(
        dataset,
        batch_size = 8,      # smaller batch size
        shuffle=True,
        num_workers=2,     # parallel loading
        collate_fn=lambda b: collate_fn(b, processor, action_tokenizer)
    )
    

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr= 5e-4 # lower learning rate   # it was -2e-4, but the original code used 5e-4
    )
    
    print("Starting training...")
    start_time = time.time()
    print(f"starting training at {start_time}")
    model.train()
    
    for epoch in range(7):
        total_loss = 0
        total_l1_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(torch.bfloat16).to(device)
            labels = batch['labels'].to(device)
            
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,  # only action tokens!
                )
                loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            action_l1_loss = torch.tensor(0.0)

            with torch.no_grad():
                action_logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab)
                action_preds = action_logits.argmax(dim=-1)  # (batch, seq_len-1)
                action_gt = labels[:, 1:]                    # (batch, seq_len-1)
                
                # Make sure shapes match
                min_len = min(action_preds.shape[1], action_gt.shape[1])
                action_preds = action_preds[:, :min_len]
                action_gt = action_gt[:, :min_len]
                
                mask = action_gt > action_tokenizer.action_token_begin_idx
                
                Q01 = np.array([-0.2737, -0.2854, -0.8444, -0.0562, -0.0814, -0.1755, -1.0])
                Q99 = np.array([ 0.7964,  0.3604,  1.0,     0.0558,  0.1694,  0.2551,  1.0])

                if mask.sum() > 0:
                    pred_normalized = action_tokenizer.decode_token_ids_to_actions(
                        action_preds[mask].cpu().numpy()
                    )
                    gt_normalized = action_tokenizer.decode_token_ids_to_actions(
                        action_gt[mask].cpu().numpy()
                    )
                    
                    pred_normalized = torch.tensor(pred_normalized)  # shape (N,)
                    gt_normalized = torch.tensor(gt_normalized)      # shape (N,)
                    
                    # Reshape to (batch*steps, 7) for denormalization
                    n_steps = len(pred_normalized) // 7
                    pred_normalized = pred_normalized.reshape(n_steps, 7)
                    gt_normalized = gt_normalized.reshape(n_steps, 7)
                    
                    Q01_t = torch.tensor(Q01, dtype=torch.float32)
                    Q99_t = torch.tensor(Q99, dtype=torch.float32)
                    
                    pred_real = (pred_normalized + 1) / 2 * (Q99_t - Q01_t) + Q01_t
                    gt_real = (gt_normalized + 1) / 2 * (Q99_t - Q01_t) + Q01_t
                    
                    action_l1_loss = torch.nn.functional.l1_loss(pred_real, gt_real)
                    #action_l1_loss = torch.nn.functional.l1_loss(
                    #    continuous_actions_pred,
                    #    continuous_actions_gt
                    #)
                    
                    total_l1_loss += action_l1_loss

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"L1: {action_l1_loss.item():.4f} | "  # ← add this
                      f"Time Elapsed: {time.time() - start_time:.2f}s")  

        avg_loss = total_loss / len(dataloader)
        average_l1_loss = total_l1_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | Avg L1: {average_l1_loss.item():.4f}") 
        model.save_pretrained(f'/code/checkpoints/epoch_{epoch+1}')
        print(f"Checkpoint saved!")

if __name__ == "__main__":
    train()