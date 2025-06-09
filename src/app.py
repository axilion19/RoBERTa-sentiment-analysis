import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_id="axilion/RoBERTa-movie-sentiment-analyzer", max_len=512):
        self.model = RobertaForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
        self.model.eval()
        self.max_len = max_len

    def predict(self, reviews):
        if isinstance(reviews, str):
            reviews = [reviews]

        results = []
        with torch.no_grad():
            enc = self.tokenizer(
                reviews,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

            logits = self.model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1).cpu().numpy()

            for idx, review in enumerate(reviews):
                label = "POSITIVE" if preds[idx] == 1 else "NEGATIVE"
                score = probs[idx, preds[idx]].item()
                results.append({"review": review, "label": label, "score": round(score, 3)})

        return results
    
if __name__ == "__main__":
    analyzer= SentimentAnalyzer()
    samples = [
        "The movie was surprisingly touching and well-acted.",
        "Plot holes everywhere—what a waste of time!",
        "I fell asleep halfway through.",
        "Amazing cinematography but weak story.",
        """i'll spare you the "cinema is dead" spiel and instead elect to mention that they literally say "epic fail" like five times in this movie""", #1/5-star
        "i have no clue what the plot of this was supposed to be, but one star because the scene of ryan in the car reminiscing about his ex while listening to all too well i fear i may know that feeling a little bit all too well.", # 1/5-star
        "they added the scene where colt cries in the car while listening to all too well by taylor swift so that girls could have their own “literally me” ryan gosling moment", #4/5-star
        "the fall guy was released in dvd/bluray format today so I had to watch it again and I know what you all will say regarding the amount of times I’ve seen this film but how could I just ignore watching it on my own new physical copy when it came in the mail today… I couldn’t, bc my love for this film is too strong", #5-star
        "A clear thank-you message to stunt performers, and all those in a craft that frequently receives less recognition than it deserves, but a film that is messy—really messy. Gosling and Blunt try their hardest to lift the weak writing, and they occasionally succeed in providing ounces of charisma and charm, but their fundamental relationship felt deeply forced. It's an effort that unfortunately falls into many of the modern-day blockbuster traps—meta humour that lands less than it should and okay-ish action sequences that feel way too long (as does the entire film). There is true enthusiasm there, and don't get me wrong, there are some genuinely pretty funny and heartfelt moments (especially the ones with Gosling leading the line), but I can't help but feel slightly disappointed overall." #2,5/5 star
    ]

    for out in analyzer.predict(samples):
        print(out)