from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
import json
import os
from tqdm import tqdm
from pymilvus import CollectionSchema, connections, FieldSchema, DataType, Collection, utility
folder_path='/home/hungday/Documents/VAST/rag/processed_data'
# model embedding
model = SentenceTransformerEmbeddingFunction('mixedbread-ai/mxbai-embed-large-v1')
#load json folder
all_entities=[]
for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for article in tqdm(data, desc=f"Processing articles in {filename}", unit="article", leave=False):
                    article['loai_van_ban_vector']=model(article['loai_van_ban'])
                    article['noi_dung_vector']= model(article['noi_dung'])

                    all_entities.append({
                    'loai_van_ban_vector': article['loai_van_ban_vector'],
                    'noi_dung_vector': article['noi_dung_vector']
                })
                print({key: data[0][key] for key in data[0].keys()})


#create database
connections.connect(alias='default',host='localhost', port='19530')
fields=[
     FieldSchema(name='id',dtype=DataType.INT64,is_primary=True),
     FieldSchema(name='loai_van_ban',dtype=DataType.FLOAT_VECTOR,dim=258),
     FieldSchema(name='noi_dung',dtype=DataType.FLOAT_VECTOR,dim=258)
]

schema= CollectionSchema(fields,description='law vectorize')

collection_name = "law_data"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
else:
    collection = Collection(name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
#insert database
insert_result=Collection.insert(all_entities)
print(f'data inserted,number of rows:{len(insert_result.primary_key)}')

index_params={
    'index_type': "IVF_FLAT",
    'metric_type':"L2",
    "param":{'nlist':100}
}
#create index for database
collection.create_index(field_name='loai_van_ban_vector',index_params=index_params)
print('index created on loai_van_ban ')
collection.create_index(field_name='noi_dung_vector',index_params=index_params)
print('index created on noi_dung_vector')
#load database to mem
collection.load()
print('loaded collection into memory') 