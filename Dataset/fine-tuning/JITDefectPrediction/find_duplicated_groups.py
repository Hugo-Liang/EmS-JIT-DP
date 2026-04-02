import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import pickle
from copy import deepcopy
import random


torch.set_printoptions(precision=4)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def load_codebert_model(device):
    # Load pre-trained CodeBERT model
    # model_name = "./models/codebert_base"
    model_name = "./models/roberta_base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)
    return tokenizer, model


def generate_embeddings(model, tokenizer, texts, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Generating embeddings'):
        batch_texts = texts[i:i + batch_size]
        # Tokenize input texts
        # inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)    # Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
        inputs_len = (inputs['attention_mask'] != 0).sum(1)
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)   # batch_size * sequence_length * hidden_size

        # Extract embeddings
        batch_embeddings = []
        for idx, input_len in enumerate(inputs_len):
            embedding = outputs.last_hidden_state[idx, 1:input_len - 1]
            embedding_polling = torch.mean(embedding, dim=0)
            embedding_normalized = F.normalize(embedding_polling, p=2, dim=0)
            batch_embeddings.append(embedding_normalized)  # Mean pooling
            # batch_embeddings.append(torch.mean(outputs.last_hidden_state[idx, 1:input_len-1], dim=0))  # Mean pooling
        batch_embeddings = torch.stack(batch_embeddings)
        embeddings.extend(batch_embeddings)
    embeddings = torch.stack(embeddings)
    return embeddings


if __name__ == '__main__':
    set_seeds(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained CodeBERT model
    tokenizer, model = load_codebert_model(device)

    # Load data
    # changes_all = torch.load('./data/jitfine/changes_all.pkl')
    for project in ['gerrit', 'go', 'jdt', 'openstack', 'platform', 'qt']:
    # for project in ['gerrit']:
        print(f'Begin to process {project}')
        with open(f'./data_SimCom/combined/{project}_all.pkl', "rb") as changes_src:
            changes_all = pickle.load(changes_src)
        changes_all_copy = deepcopy(changes_all)
        for i, commit_change in tqdm(enumerate(changes_all_copy[3]), desc='Combine code changes'):
            commit_changes = ''
            for code in commit_change:
                commit_changes += f"{code} "
            changes_all[3][i] = commit_changes.strip()
            # if i == 6:
            #     print(commit_change)
            #     print(changes_all[3][i])
            # print(commit_changes)
            #     break

        # Check if embeddings file already exists
        if not os.access('./data_SimCom/retrieved/', os.F_OK):
            os.mkdir('./data_SimCom/retrieved/')
        embeddings_file = f'./data_SimCom/retrieved/embeddings_{project}.pt'
        if not os.path.isfile(embeddings_file):
            # Generate embeddings for all changes and save to file
            all_embeddings = generate_embeddings(model, tokenizer, changes_all[3], device)
            torch.save(all_embeddings, embeddings_file)
        else:
            # Load embeddings from file
            all_embeddings = torch.load(embeddings_file).to(device)

            # 随机采样一部分文本，并使用相同的方法生成嵌入表示
            num_samples = 1000  # 采样的文本数量
            random_indices = random.sample(range(len(changes_all[3])), num_samples)  # 随机选取的索引
            random_texts = [changes_all[3][idx] for idx in random_indices]  # 随机选取的文本
            random_embeddings = all_embeddings[random_indices]  # 随机选取的嵌入表示
            generated_embeddings = generate_embeddings(model, tokenizer, random_texts, device)

            # 对比新生成的嵌入表示与从文件中加载的嵌入表示
            if torch.allclose(random_embeddings, generated_embeddings, atol=1e-4):
                print("加载的嵌入表示与原始数据集匹配，并且没有乱序。")
            else:
                print("加载的嵌入表示与原始数据集不匹配，或者存在乱序。")

        # 释放GPU显存
        torch.cuda.empty_cache()

        results = []
        results_filtered = []

        # Set batch size for processing
        batch_size = 16
        num_batches = len(changes_all[0]) // batch_size + 1

        # Iterate over each batch of query changes
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(changes_all[0]))

            # Get query embeddings for this batch
            query_embeddings = all_embeddings[start_idx:end_idx]
            query_changes = changes_all[3][start_idx:end_idx]
            query_hashes = changes_all[0][start_idx:end_idx]
            # for idx, hash in enumerate(query_hashes):
            #     if hash == 'bd6143c4db26626ba7570c77d2561cd1aab35a2c':  #
            #         print('Notice')

            query_labels = changes_all[1][start_idx:end_idx]
            query_messages = changes_all[2][start_idx:end_idx]

            # Compute cosine similarity between query embeddings and all embeddings
            cosine_sim_matrix = torch.cosine_similarity(query_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=-1)

            # Find top-k similar changes for each query change
            # k = 10  # Specify the number of similar changes to find
            for i in range(len(query_changes)):
                # if query_hashes[i] == '88f5e11d8bf820b0124be0f6ec3c2d96011592d9':
                # if query_hashes[i] == '2b1d413ee90dfe2e9ae41c35ab37253df53fc6cd':
                # Specific scenario: query hashA(88f5e11d8bf820b0124be0f6ec3c2d96011592d9) has three matched samples:
                # hashB(2b1d413ee90dfe2e9ae41c35ab37253df53fc6cd)
                # hashC(adb0ac4e5454391d68026cbeee93169578a10743)
                # hashD(559bf87fd0d92e4d230058f5819c78f8b727d326)
                # where hashB is the exact clone sample, hashC and hashD are the filtered clone samples
                # Therefore, [hashA and hashB] are saved to the similar_changes_results_openstack.json
                # [hashA, hashC, and hashD], as well as [hashB, hashC, and hashD] are saved to the similar_changes_results_openstack_filtered.json
                #     break

                # _, topk_indices = torch.topk(cosine_sim_matrix[i], k=k+1)
                # topk_indices = torch.where(cosine_sim_matrix[i] == 1)[0]  # failed to find similarity results for some samples due to the issue of precision loss between GPU and CP. i.e. bd6143c4db26626ba7570c77d2561cd1aab35a2c-97a54ac2801e24d63a10eaabba108007562c74d3(idx: 775-788)
                # topk_indices = torch.where(cosine_sim_matrix[i] >= 0.99999)[0]    # fcec2cda9067af1594f32d0d9c31eb06e816214b-96d1b9710b43ffb1cd79d862380319f30c8797c2 3385-3390 sim=0.9999
                # tensor_one = torch.ones(1).to('cuda') # failed to find similarity results for some samples due to the issue of precision loss between GPU and CP. i.e. bd6143c4db26626ba7570c77d2561cd1aab35a2c-97a54ac2801e24d63a10eaabba108007562c74d3(idx: 775-788)
                tensor_one = cosine_sim_matrix[i][start_idx+i]
                # tensor_one_float32 = torch.tensor(1.0, dtype=torch.float32, device='cuda:0')
                # tensor_one_float64 = torch.tensor(1.0, dtype=torch.float64, device='cuda:0')

                # topk_indices = torch.where(torch.isclose(cosine_sim_matrix[i], tensor_one, atol=1e-5))[0]

                topk_indices = torch.where(cosine_sim_matrix[i] == tensor_one)[0]
                topk_indices = topk_indices[topk_indices != start_idx+i]  # Remove the query change itself from the top-k indices

                # Compare code changes to ensure exact match
                exact_matches = []
                filtered_candidates = []
                for idx in topk_indices:
                    sim_tensor = cosine_sim_matrix[i, idx]
                    try:
                        sim_float = eval(str(sim_tensor).split('(')[1].split(',')[0])
                    except:
                        print(f"string format of sim_tensor is:{str(sim_tensor)}")
                    if query_changes[i] == changes_all[3][idx]:
                        exact_matches.append((sim_float, changes_all[0][idx], changes_all[1][idx], changes_all[2][idx], changes_all[3][idx]))
                    else:
                        filtered_candidates.append((sim_float, changes_all[0][idx], changes_all[1][idx],
                                               changes_all[2][idx], changes_all[3][idx]))

                # Save results
                if len(exact_matches) != 0:
                    result = {
                        "query_hash": query_hashes[i],
                        "query_label": query_labels[i],
                        "query_message": query_messages[i],
                        "query_change": query_changes[i],
                        "similar_changes": [{"similarity": sim, "hash": hash_value, "label": label, "message": message, "change": change} for
                                            sim, hash_value, label, message, change in exact_matches]
                    }
                    results.append(result)
                if len(filtered_candidates) != 0:
                    result_filtered = {
                        "query_hash": query_hashes[i],
                        "query_label": query_labels[i],
                        "query_message": query_messages[i],
                        "query_change": query_changes[i],
                        "similar_changes": [
                            {"similarity": sim, "hash": hash_value, "label": label, "message": message, "change": change} for
                            sim, hash_value, label, message, change in filtered_candidates]
                    }
                    results_filtered.append(result_filtered)
        if not os.access(f'./data_SimCom/clone/{project}/', os.F_OK):
            os.mkdir(f'./data_SimCom/clone/{project}/')

        # Save results to JSON file
        with open(f'./data_SimCom/clone/{project}/similar_changes_results_{project}.json', 'w') as f, open(f'./data_SimCom/clone/{project}/similar_changes_results_{project}_filtered.json', 'w') as f_filtered:
            json.dump(results, f, indent=4)
            json.dump(results_filtered, f_filtered, indent=4)
        print('-' * 50)
