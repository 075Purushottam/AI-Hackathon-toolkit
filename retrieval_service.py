from flask import Flask, request, jsonify
import os, sys, ast, re
import pandas as pd
from langchain_community.vectorstores import FAISS
from openai import AzureOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config, json, pickle, numpy as np
from langchain_openai import AzureOpenAIEmbeddings
import tiktoken

app = Flask(__name__)

MODEL_FOLDER = "models"
MAPPING_FILE = "model_mapping.xlsx"

def llm_client():
    client = AzureOpenAI(
            api_key=config.API_KEY,
            api_version=config.API_VERSION,
            azure_endpoint=config.BASE_URL,
        )
    return(client)

def llm_call(prompt, max_tokens, temperature):
    client = llm_client()
    completion = client.chat.completions.create(
        model=config.DEPOLYMENT_NAME,
        messages=[
            {
                'role': 'user', 
                'content': prompt
            }
        ],
        max_tokens=max_tokens,
        temperature = temperature
    )
    #print(text_token_size(completion))
    return completion.choices[0].message.content

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\2725939\Downloads\reranker_models\colbert")
# model = AutoModel.from_pretrained(r"C:\Users\2725939\Downloads\reranker_models\colbert")

# def encode(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)  # [CLS] pooling

# def rerank(query,retrieved_contexts,top_m):
#     # Encode query
#     query_embedding = encode(query)

#     # Re-rank contexts
#     scores = []
#     for context in retrieved_contexts:
#         context_embedding = encode(context)
#         score = F.cosine_similarity(query_embedding, context_embedding).item()
#         scores.append((context, score))

#     # Sort by score
#     ranked_contexts = sorted(scores, key=lambda x: x[1], reverse=True)

#     # Top-N for LLM
#     top_contexts = [ctx for ctx, _ in ranked_contexts[:5]]

#     return top_contexts

def generate_clusters(corpus):
    
    max_queries = config.CLUSTER_MAX_QUERY
    n_clusters = len(corpus)/max_queries

    prompt = f"""
    You are an intelligent cluster generator.
    Create {n_clusters} for the INPUT queries by understanding semantic similarity 
    between each sentences. Each cluster will have maximum {max_queries} values.
    
    INSTRUCTIONS:
    1. Do not include any additional text, explanations, or markdown formatting.
    2. Do not assume or generate data not explicitly visible.

    INPUT:
    {corpus}
    
    OUTPUT: 
    Provide the output in list of lists where each inner list will be values inside the cluster.    
    """
    
    value = llm_call(prompt, config.MAX_TOKENS, config.TEMPERATURE)
    return(value)
# Utility: Load model paths from Excel mapping
def get_model_paths(pdf_filename):
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError("Model mapping file not found.")
    
    df = pd.read_excel(MAPPING_FILE)
    row = df[df["PDF Filename"] == pdf_filename]
    # print(row)
    if row.empty:
        raise ValueError(f"No models found for {pdf_filename}")
    
    faiss_path = row.iloc[0]["FAISS Path"]
    bm25_path = row.iloc[0]["BM25 Path"]
    return faiss_path, bm25_path

def load_faiss(faiss_path):
    embeddings = AzureOpenAIEmbeddings(
        model=config.emd_model,
        azure_endpoint = config.emd_endpoint,
        api_key=config.emd_api_key,
        api_version=config.emd_api_version
    )
    return FAISS.load_local(faiss_path,embeddings,allow_dangerous_deserialization=True)

def query_faiss(vectorstore,query,top_k):
    result = vectorstore.similarity_search(query,k=top_k)
    print("Query vectors")
    for i, res in enumerate(result,start=1):
        print(f"[{i}] {res}\n")
    return result


def query_faiss_with_neighbors(vectorstore, corpus, query, top_k=2, prev_ct=1, next_ct=1):

    results = vectorstore.similarity_search(query, k=top_k)

    faiss_results = []
    for res in results:
        chunk_id = res.metadata.get("chunk_id")
        if chunk_id is None:
            raise ValueError("chunk_id not found in metadata. Ensure you added it during indexing.")

        # Slice neighbors from corpus using chunk_id
        start = max(0, chunk_id - prev_ct)
        end = min(len(corpus), chunk_id + next_ct + 1)
        context_chunks = corpus[start:end]

        # Build structured result
        faiss_results.append({
            "context": [
                {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                } for chunk in context_chunks
            ]
        })

    # print("Result from FAISS:\n")    
    # for i, res in enumerate(faiss_results,start=1):
    #     print(f"[{i}] {res}\n")
        
    return faiss_results


def load_bm25(bm25_path):
    with open(bm25_path,"rb") as f:
        data = pickle.load(f)
    return data['bm25'],data['corpus_tokens']

def query_bm25(query, bm25, corpus, top_k):
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[::-1][:top_k]

    bm25_results = []
    for i in top_n:
        # print(corpus[i].metadata,"\n")
        bm25_results.append({
            "page_content": corpus[i].page_content,
            "metadata": {
                "bm25_score": bm25_scores[i],
                "page_number": corpus[i].metadata.get("page"),
                "chunk_id": corpus[i].metadata.get("chunk_id")
            }
        })
    print("BM25 contexts:")
    for i, res in enumerate(bm25_results,start=1):
        print(f"[{i}] {res}\n")
    return bm25_results

def query_bm25_with_neighbour(query,bm25,corpus,top_k,prev_ct,next_ct):
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query) 
    top_n = np.argsort(bm25_scores)[::-1][:top_k] 
    bm25_results = [{"page_content": corpus[i-prev_ct:i+next_ct+1]} for i in top_n]
    print("Result from BM25:\n")
    for i, res in enumerate(bm25_results,start=1):
        print(f"[{i}] {res}\n")
    return bm25_results

def ask_with_azure(query, contexts):
    prompt = f'''
        You are a financial document analysis assistant.
        Your task is to answer multiple user queries based on the provided context extracted from financial documents.

        ### Queries:
        {query}

        ### Context:
        {contexts}

        ### Instructions:
        - Each sub-query in "query" is prefixed with a sequence number (e.g., 0., 1., 2., 3.).
        - Use only the information in the context to answer. Do not generate and assume any details beyond the context.
        
        ### Output:
        - For each sub-query, return the answer as JSON where:
            - Key = sequence number
            - Value = In Value give me JSON with this three keys 'answer':'answer text','page_no':'from which page you get the answer', 'confidence': 'provide me the confidence score of answer between (0-1) in integer' , “Answer only if the information is explicitly present in the provided context; \n even if found the answer but it is partial then in this case give me 'Not found' ' if not found, respond with 'Not found' ONLY as Value for both answer and page_no.\n  Do not infer or generate any additional details.”
        - Do not include any explanation or extra text outside the JSON.
    '''
    response = llm_client().chat.completions.create(
        model=config.DEPOLYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        
        response = response.choices[0].message.content
        response = response.replace("`", "").replace("'", "\'")
        response = re.sub(r'^(json|python)\s*', '', response, flags=re.IGNORECASE)
        
        # response = re.sub(r'\\x[0-9A-Fa-f]{2}', '', response)
        # response = response.replace("\\", "\\\\")

        if response.startswith("json"): 
            response = response[4:]
        if response.startswith("python"):
            response = response[6:]
        print("response: ",response)
        parsed_response = json.loads(response)
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens_input = encoding.encode(prompt)
        tokens_output = encoding.encode(response)
    
        print("INPUT TOKEN: ",len(tokens_input),"\n")
        print("OUTPUT TOKEN: ",len(tokens_output),"\n")

    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)

    return parsed_response,len(tokens_input),len(tokens_output)


def execute_clusters(cluster_queries,vectorstore,bm25,corpus):
    
    result = []
    all_not_found_queries = []
    overall_token_used=0

    for key,queries in cluster_queries.items():
        total_context_bucket=[]
        query_for_llm=""

        for i,query in enumerate(queries):
            faiss_context = query_faiss(vectorstore,query,config.DEFAULT_TOP_K_FAISS)
            bm25_context = query_bm25(query,bm25,corpus,config.DEFAULT_TOP_K_BM25)
            total_context_bucket.extend(faiss_context)
            total_context_bucket.extend(bm25_context)
            query_for_llm+=f"{i}. {query} \n"
        
        print(query_for_llm,"\n")
        print(f"Total context utilized for {key} is {len(total_context_bucket)}.\n")
        context_limit=config.CONTEXT_LIMIT
        cluster_result=[]
        not_found_queries = []
        tot_inp,tot_out=0,0
        for i in range(0, len(total_context_bucket), context_limit):
        
            print(f"Execution start for context:  {i} to {i+context_limit}\n")
            response,input_tokens,output_tokens = ask_with_azure(query_for_llm,total_context_bucket[i:i + context_limit])
            tot_inp+=input_tokens
            tot_out+=output_tokens
            print("Response:",response)
        
            found_queries = ([{queries[int(k)]: v} for k, v in response.items() if v['answer'] != "Not found"])
            not_found_queries = ([{k: queries[int(k)]} for k, v in response.items() if v['answer'] == "Not found"])
            cluster_result.extend(found_queries)
            query_for_llm = "\n".join([f"{k}. {queries[int(k)]}" for k, v in response.items() if v['answer'] == "Not found"])
            
            for k, v in response.items():
                print(k,{queries[int(k)]:v})

            print("cluster length: ",len(cluster_result),"\n")
            if len(query_for_llm) == 0:
                print("There is no queries left to determine move to next cluster.\n")
                break
        
        print(f"For Cluster: {key} \n")
        if len(query_for_llm) != 0:
            print("NOT FOUND Queries:\n",not_found_queries)
            all_not_found_queries.extend(not_found_queries)

        print("Found Queries with Result:\n",cluster_result,"\n")
        overall_token_used+=(tot_inp)+(tot_out)
        result.extend(cluster_result)
        with open("cluster_result_sicav_pdf_6.json", "a") as json_file:
            json.dump(cluster_result, json_file, indent=4)
    print("Total token used for found query: ",overall_token_used)
    return result,all_not_found_queries,overall_token_used

def execute_for_not_found(queries,vectorstore,bm25,corpus):
    result = []
    total_token_used=0
    for not_found_queries in queries:
        for k,query in not_found_queries.items():
            faiss_context = query_faiss_with_neighbors(vectorstore,corpus,query,config.ITERATIVE_TOP_K_FAISS,config.PREVIOUS_CONTEXT_COUNT,config.NEXT_CONTEXT_COUNT)
            bm25_context = query_bm25_with_neighbour(query,bm25,corpus,config.ITERATIVE_TOP_K_BM25,config.PREVIOUS_CONTEXT_COUNT,config.NEXT_CONTEXT_COUNT)
            _query = query
            query = f"{k}. {query}"
            response,inp_tok,out_tok = ask_with_azure(query,faiss_context+bm25_context)
            print(f"{_query}: {response}\n")
            found_queries = ([{_query: v} for k, v in response.items()])
            result.extend(found_queries)
            total_token_used+=inp_tok+out_tok
    print("Total token used for not found queries:",total_token_used)
    return result,total_token_used

@app.route("/retrieve_cluster", methods=['POST'])
def retrieve_cluster():
    try:
        # 1. Get input data
        # data = request.get_json()
        pdf_filename = request.form.get("pdf_filename")
        print(pdf_filename)
        queries = request.form.get("queries")  # Expecting a list of queries
        queries = json.loads(queries)
        print("fsdf")
        cluster_queries={}
        if len(queries)>config.CLUSTER_MAX_QUERY:
            value = ast.literal_eval(generate_clusters(queries))
            print(value)
            t1 = list(map(lambda x : "cluster_" + str(x),range(1, len(value)+1)))
            cluster_queries = dict(zip(t1, value))
        else:
            print("fdssd")
            # cluster_queries = dict(zip(list(map(lambda x : "Query_" + str(x+1), range(len(queries))), queries)))
            cluster_queries={
                "Cluster_1":queries
            }
        #     cluster_queries=request.form.get("cluster_queries")
        #     cluster_queries = json.loads(cluster_queries)
        print("Cluster Queries",cluster_queries)

        if not pdf_filename or not queries:
            return jsonify({"status": "error", "message": "pdf_filename and queries are required"}), 400

        # 2. Load model paths from mapping
        faiss_path, bm25_path = get_model_paths(pdf_filename)
        print(faiss_path)
        print(bm25_path)
        # 3. Load FAISS and BM25 models
        vectorstore = load_faiss(faiss_path)  
        bm25, corpus = load_bm25(bm25_path)   
        print("Models Loaded")

        results = [] 
        # 4. Process cluster queries
        found_query_result,all_not_found_queries,total_tokens_1 = execute_clusters(cluster_queries,vectorstore,bm25,corpus)

        # 5. Execute not found queries for second cycle 
        not_found_query_result,total_tokens_2 = execute_for_not_found(all_not_found_queries,vectorstore,bm25,corpus)
        results.extend(found_query_result)
        results.extend(not_found_query_result)
        results.append({
            "total token consumption":total_tokens_1+total_tokens_2
        })

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8078)
