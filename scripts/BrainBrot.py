# Brain Rot Chatbot Training Pipeline
import praw
import json
import re
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    TextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from datasets import Dataset

# Step 1: Data Collection from Reddit
class BrainRotScraper:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Target subreddits for maximum brain rot
        self.target_subreddits = [
            'okbuddyretard',
            'dankmemes', 
            'memes',
            'comedyheaven',
            'copypasta',
            'teenagers',
            'GenZ',
            'shitposting',
            'deepfriedmemes',
            'surrealmemes'
        ]
    
    def scrape_brain_rot(self, posts_per_sub=2000):
        """Scrape brain rot content from Reddit with proper rate limiting for large datasets"""
        import time
        import pickle
        import os
        
        all_text = []
        checkpoint_file = "scraping_checkpoint.pkl"
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                all_text = checkpoint_data.get('all_text', [])
                completed_subs = checkpoint_data.get('completed_subs', [])
                print(f"Resuming from checkpoint: {len(all_text)} texts collected")
        else:
            completed_subs = []
        
        for subreddit_name in self.target_subreddits:
            if subreddit_name in completed_subs:
                print(f"Skipping r/{subreddit_name} (already completed)")
                continue
                
            print(f"Scraping r/{subreddit_name}...")
            subreddit = self.reddit.subreddit(subreddit_name)
            
            post_count = 0
            batch_texts = []
            
            try:
                # Get posts from multiple time periods for more data
                post_sources = [
                    subreddit.hot(limit=posts_per_sub//3),
                    subreddit.top('month', limit=posts_per_sub//3),
                    subreddit.new(limit=posts_per_sub//3)
                ]
                
                for post_source in post_sources:
                    for submission in post_source:
                        try:
                            # Skip if we've seen this post before
                            if submission.id in [t.get('id') for t in batch_texts]:
                                continue
                                
                            # Add post title and text
                            text_content = f"{submission.title}"
                            if submission.selftext:
                                text_content += f" {submission.selftext}"
                            
                            # Get comments with better rate limiting
                            try:
                                submission.comments.replace_more(limit=0)
                                comments = submission.comments.list()[:8]  # Top 8 comments
                                
                                for comment in comments:
                                    if (hasattr(comment, 'body') and 
                                        len(comment.body) > 10 and 
                                        comment.score > 0 and
                                        not comment.body.startswith('[deleted]')):
                                        text_content += f" {comment.body}"
                                        
                            except Exception as comment_error:
                                # Skip comments if there's an issue, keep the post
                                pass
                            
                            batch_texts.append({
                                'id': submission.id,
                                'text': text_content,
                                'subreddit': subreddit_name,
                                'score': submission.score
                            })
                            
                            post_count += 1
                            
                            # Adaptive rate limiting based on API response
                            if post_count % 5 == 0:
                                time.sleep(1)  # Short pause every 5 posts
                                
                            if post_count % 25 == 0:
                                print(f"  Processed {post_count} posts from r/{subreddit_name}")
                                time.sleep(3)  # Longer pause every 25 posts
                                
                            if post_count % 100 == 0:
                                # Save checkpoint every 100 posts
                                all_text.extend([item['text'] for item in batch_texts])
                                checkpoint_data = {
                                    'all_text': all_text,
                                    'completed_subs': completed_subs
                                }
                                with open(checkpoint_file, 'wb') as f:
                                    pickle.dump(checkpoint_data, f)
                                print(f"  Checkpoint saved: {len(all_text)} total texts")
                                batch_texts = []
                                
                        except Exception as post_error:
                            print(f"  Error processing post: {post_error}")
                            time.sleep(2)  # Longer pause on errors
                            continue
                            
            except Exception as sub_error:
                print(f"  Error accessing r/{subreddit_name}: {sub_error}")
                print("  Waiting 30 seconds before continuing...")
                time.sleep(30)
                continue
            
            # Add remaining batch texts
            all_text.extend([item['text'] for item in batch_texts])
            completed_subs.append(subreddit_name)
            
            print(f"  Completed r/{subreddit_name}: {post_count} posts")
            print(f"  Total texts collected so far: {len(all_text)}")
            
            # Save checkpoint after each subreddit
            checkpoint_data = {
                'all_text': all_text,
                'completed_subs': completed_subs
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Longer pause between subreddits
            print("  Waiting 10 seconds before next subreddit...")
            time.sleep(10)
        
        # Clean up checkpoint file when done
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            
        print(f"Scraping complete! Collected {len(all_text)} total texts.")
        return all_text
    
    def clean_text(self, texts):
        """Clean and filter the scraped text"""
        cleaned = []
        for text in texts:
            # Remove URLs, markdown formatting, etc.
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\*+', '', text)
            text = re.sub(r'&gt;', '>', text)
            text = re.sub(r'&lt;', '<', text)
            
            # Filter out very short or very long texts
            if 10 < len(text) < 500:
                cleaned.append(text.strip())
        
        return cleaned

# Step 2: Model Configuration and Training
class BrainRotTrainer:
    def __init__(self):
        # Small GPT-2 config for faster training
        self.config = GPT2Config(
            vocab_size=50257,
            n_positions=512,      # Shorter context for memes
            n_embd=512,          # Smaller embedding size
            n_layer=8,           # Fewer layers
            n_head=8,            # Fewer attention heads
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel(self.config)
        
    def prepare_dataset(self, texts):
        """Prepare the brain rot dataset for training"""
        # Format texts for conversation-style training
        formatted_texts = []
        for text in texts:
            # Add special tokens to help model learn conversational patterns
            formatted_text = f"<|startoftext|>{text}<|endoftext|>"
            formatted_texts.append(formatted_text)
        
        # Tokenize the texts
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
        
        return dataset
    
    def train_model(self, dataset, output_dir="./brainrot-gpt2"):
        """Train the brain rot model"""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            prediction_loss_only=True,
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )
        
        # Split dataset for training/validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

# Step 3: Brain Rot Chatbot Interface
class BrainRotChatbot:
    def __init__(self, model_path="./brainrot-gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
    
    def generate_response(self, prompt, max_length=100, temperature=0.8):
        """Generate a brain rot response"""
        # Encode the prompt
        input_text = f"<|startoftext|>{prompt}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(input_text.replace("<|startoftext|>", ""), "").strip()
        
        return response
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("Brain Rot Chatbot activated! Type 'quit' to exit.")
        print("Prepare for maximum brain rot...")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            
            response = self.generate_response(user_input)
            print(f"Brain Rot Bot: {response}")

# Usage Example
if __name__ == "__main__":
    # Step 1: Scrape data (you'll need Reddit API credentials)
    
    scraper = BrainRotScraper(
        client_id="k4RKEdOhxfljGi0psvkg2w",
        client_secret="5fbHWbaT_h1USXTHH3ZqR1Kykv86MA", 
        user_agent="BrainRotBot/1.0 by Previous_Impress5216"
    )
    
    print("Scraping brain rot content...")
    raw_texts = scraper.scrape_brain_rot(posts_per_sub=500)
    clean_texts = scraper.clean_text(raw_texts)
    
    # Save the data
    with open('brain_rot_data.json', 'w') as f:
        json.dump(clean_texts, f)
    
    
    # Step 2: Train the model (assuming you have the data)
    """
    trainer = BrainRotTrainer()
    
    # Load your scraped data
    with open('brain_rot_data.json', 'r') as f:
        brain_rot_texts = json.load(f)
    
    print("Preparing dataset...")
    dataset = trainer.prepare_dataset(brain_rot_texts)
    
    print("Training brain rot model...")
    trainer.train_model(dataset)
    """
    
    # Step 3: Use the trained chatbot
    print("Loading brain rot chatbot...")
    chatbot = BrainRotChatbot()
    chatbot.chat_loop()