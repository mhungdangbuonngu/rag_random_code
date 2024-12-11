from sentence_transformers import SentenceTransformer
dimensions = 258
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

text='anh yeu em'
print(text,'\n')
embed_text=model.encode(text)
print(embed_text)
