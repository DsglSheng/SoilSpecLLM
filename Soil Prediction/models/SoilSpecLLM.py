from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, T5Config, T5EncoderModel, T5Tokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len 
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BGE-M3':
            self.bge_config = XLMRobertaConfig.from_pretrained('BAAI/bge-m3')

            self.bge_config.num_hidden_layers = configs.llm_layers
            self.bge_config.output_attentions = True
            self.bge_config.output_hidden_states = True
            try:
                self.llm_model = XLMRobertaModel.from_pretrained(
                    'BAAI/bge-m3',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bge_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = XLMRobertaModel.from_pretrained(
                    'BAAI/bge-m3',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bge_config,
                )

            try:
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                    'BAAI/bge-m3',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                    'BAAI/bge-m3',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'T5':
            self.t5_config = T5Config.from_pretrained('t5-base')

            self.t5_config.num_layers = configs.llm_layers
            self.t5_config.output_attentions = True
            self.t5_config.output_hidden_states = True
            try:
                self.llm_model = T5EncoderModel.from_pretrained(
                    't5-base',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.t5_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = T5EncoderModel.from_pretrained(
                    't5-base',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.t5_config,
                )

            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    't5-base',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = T5Tokenizer.from_pretrained(
                    't5-base',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')
        # B鏄壒娆″ぇ灏忥紝T鏄椂闂村簭鍒楅暱搴︼紝N鏄彉閲忔暟閲?        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 璁＄畻浜嗘椂闂村簭鍒楃殑鍏抽敭缁熻鐗瑰緛锛氭渶灏忓€硷紝鏈€澶у€硷紝涓綅鏁帮紝鑷浉鍏虫粸鍚庡€硷紙閫氳繃calcute_lags鏂规硶锛夛紝瓒嬪娍锛堥€氳繃璁＄畻宸垎鍜屽垽鏂璐燂級
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        # 鎭㈠鍘熷缁村害
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # 灏嗘彁绀鸿浆鎹负token骞惰幏鍙栧祵鍏?        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # 鑾峰彇婧愬祵鍏?        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # 琛ヤ竵宓屽叆
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        # 閫氳繃閲嶇紪绋嬪眰澶勭悊宓屽叆
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        # 閫氳繃閲嶇紪绋嬪眰澶勭悊宓屽叆
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    # 浣跨敤FFT锛堝揩閫熷倕閲屽彾鍙樻崲锛夎绠楄嚜鐩稿叧鍑芥暟锛岀劧鍚庢彁鍙杢op-k鐨勬粸鍚庡€?    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    # 鏍稿績鍒涙柊鐐癸紝鏂囨湰涓庢椂闂村簭鍒楀榻?    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        # 灏嗘椂闂村簭鍒楃壒寰佷笌LLM鐨勮瘝宓屽叆绌洪棿瀵归綈

        # b: batch size锛堟壒娆″ぇ灏忥級
        # l: length锛堝簭鍒楅暱搴︼紝杩欓噷鏄洰鏍囧祵鍏ョ殑闀垮害锛?        # h: heads锛堟敞鎰忓姏澶存暟閲忥級
        # e: embedding dimension锛堟瘡涓ご鐨勫祵鍏ョ淮搴︼級
        # s: source length锛堟簮宓屽叆鐨勯暱搴︼級
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        # 璁＄畻娉ㄦ剰鍔涘垎鏁?        # 灏?target_embedding (褰㈢姸涓?[batch_size, target_length, num_heads, head_dim]) 涓?source_embedding (褰㈢姸涓?[source_length, num_heads, head_dim]) 杩涜鐭╅樀涔樻硶
        # 鍏蜂綋鏉ヨ锛屽畠瀵筫缁村害杩涜姹傚拰锛岃繖鏄洜涓篹鍦ㄨ緭鍑轰腑娑堝け浜嗭紝琛ㄧず鍦ㄨ繖涓淮搴︿笂杩涜鐐圭Н鎿嶄綔
        # 鏈€缁堝緱鍒板舰鐘朵负[batch_size, num_heads, target_length, source_length]鐨勬敞鎰忓姏鍒嗘暟鐭╅樀
        # 杩欎釜鎿嶄綔鏈川涓婃槸鍦ㄨ绠楁敞鎰忓姏鏈哄埗涓殑鏌ヨ(query)鍜岄敭(key)涔嬮棿鐨勭浉浼煎害鍒嗘暟銆傚湪鑷敞鎰忓姏鏈哄埗涓紝杩欎竴姝ラ鐢ㄤ簬纭畾搴忓垪涓笉鍚屼綅缃箣闂寸殑鍏宠仈寮哄害銆?        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 璁＄畻鍔犳潈鍜?        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding




# 娑堣瀺瀹為獙
# from math import sqrt
# 
# import torch
# import torch.nn as nn
# 
# from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
#     BertModel, BertTokenizer
# from layers.Embed import PatchEmbedding
# import transformers
# from layers.StandardNorm import Normalize
# 
# transformers.logging.set_verbosity_error()
# 
# 
# class FlattenHead(nn.Module):
#     def __init__(self, n_vars, nf, target_window, head_dropout=0):
#         super().__init__()
#         self.n_vars = n_vars
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear = nn.Linear(nf, target_window)
#         self.dropout = nn.Dropout(head_dropout)
# 
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.dropout(x)
#         return x
# 
# 
# class Model(nn.Module):
# 
#     def __init__(self, configs, patch_len=16, stride=8):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.pred_len = configs.pred_len
#         self.seq_len = configs.seq_len
#         self.d_ff = configs.d_ff
#         self.top_k = 5
#         self.d_llm = configs.llm_dim
#         self.patch_len = configs.patch_len
#         self.stride = configs.stride
# 
#         if configs.llm_model == 'LLAMA':
#             # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
#             self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
#             self.llama_config.num_hidden_layers = configs.llm_layers
#             self.llama_config.output_attentions = True
#             self.llama_config.output_hidden_states = True
#             try:
#                 self.llm_model = LlamaModel.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
#                     'meta-llama/Llama-2-7b-hf',
#                     trust_remote_code=True,
#                     local_files_only=True,
#                     config=self.llama_config,
#                     load_in_4bit=True
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 self.llm_model = LlamaModel.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
#                     'meta-llama/Llama-2-7b-hf',
#                     trust_remote_code=True,
#                     local_files_only=False,
#                     config=self.llama_config,
#                     load_in_4bit=True
#                 )
#             try:
#                 self.tokenizer = LlamaTokenizer.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
#                     'meta-llama/Llama-2-7b-hf',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 self.tokenizer = LlamaTokenizer.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
#                     'meta-llama/Llama-2-7b-hf',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         elif configs.llm_model == 'GPT2':
#             self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
# 
#             self.gpt2_config.num_hidden_layers = configs.llm_layers
#             self.gpt2_config.output_attentions = True
#             self.gpt2_config.output_hidden_states = True
#             try:
#                 self.llm_model = GPT2Model.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=True,
#                     config=self.gpt2_config,
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 self.llm_model = GPT2Model.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=False,
#                     config=self.gpt2_config,
#                 )
# 
#             try:
#                 self.tokenizer = GPT2Tokenizer.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 self.tokenizer = GPT2Tokenizer.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         elif configs.llm_model == 'BERT':
#             self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
# 
#             self.bert_config.num_hidden_layers = configs.llm_layers
#             self.bert_config.output_attentions = True
#             self.bert_config.output_hidden_states = True
#             try:
#                 self.llm_model = BertModel.from_pretrained(
#                     'google-bert/bert-base-uncased',
#                     trust_remote_code=True,
#                     local_files_only=True,
#                     config=self.bert_config,
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 self.llm_model = BertModel.from_pretrained(
#                     'google-bert/bert-base-uncased',
#                     trust_remote_code=True,
#                     local_files_only=False,
#                     config=self.bert_config,
#                 )
# 
#             try:
#                 self.tokenizer = BertTokenizer.from_pretrained(
#                     'google-bert/bert-base-uncased',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 self.tokenizer = BertTokenizer.from_pretrained(
#                     'google-bert/bert-base-uncased',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         else:
#             raise Exception('LLM model is not defined')
# 
#         if self.tokenizer.eos_token:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         else:
#             pad_token = '[PAD]'
#             self.tokenizer.add_special_tokens({'pad_token': pad_token})
#             self.tokenizer.pad_token = pad_token
# 
#         for param in self.llm_model.parameters():
#             param.requires_grad = False
# 
#         if configs.prompt_domain:
#             self.description = configs.content
#         else:
#             self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
# 
#         self.dropout = nn.Dropout(configs.dropout)
# 
#         self.patch_embedding = PatchEmbedding(
#             configs.d_model, self.patch_len, self.stride, configs.dropout)
# 
#         # 鍘绘帀patch-programming鐩稿叧缁勪欢
#         # self.word_embeddings = self.llm_model.get_input_embeddings().weight
#         # self.vocab_size = self.word_embeddings.shape[0]
#         # self.num_tokens = 1000
#         # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
#         # self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
# 
#         # 鐩存帴浣跨敤绠€鍗曠殑绾挎€ф槧灏勬浛浠atch-programming
#         self.direct_mapping = nn.Linear(configs.d_model, self.d_llm)
# 
#         self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
#         self.head_nf = self.d_ff * self.patch_nums
# 
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
#                                                  head_dropout=configs.dropout)
#         else:
#             raise NotImplementedError
# 
#         self.normalize_layers = Normalize(configs.enc_in, affine=False)
# 
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]
#         return None
# 
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# 
#         x_enc = self.normalize_layers(x_enc, 'norm')
#         # B鏄壒娆″ぇ灏忥紝T鏄椂闂村簭鍒楅暱搴︼紝N鏄彉閲忔暟閲?#         B, T, N = x_enc.size()
#         x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
# 
#         # 璁＄畻浜嗘椂闂村簭鍒楃殑鍏抽敭缁熻鐗瑰緛锛氭渶灏忓€硷紝鏈€澶у€硷紝涓綅鏁帮紝鑷浉鍏虫粸鍚庡€硷紙閫氳繃calcute_lags鏂规硶锛夛紝瓒嬪娍锛堥€氳繃璁＄畻宸垎鍜屽垽鏂璐燂級
#         min_values = torch.min(x_enc, dim=1)[0]
#         max_values = torch.max(x_enc, dim=1)[0]
#         medians = torch.median(x_enc, dim=1).values
#         lags = self.calcute_lags(x_enc)
#         trends = x_enc.diff(dim=1).sum(dim=1)
# 
#         prompt = []
#         for b in range(x_enc.shape[0]):
#             min_values_str = str(min_values[b].tolist()[0])
#             max_values_str = str(max_values[b].tolist()[0])
#             median_values_str = str(medians[b].tolist()[0])
#             lags_values_str = str(lags[b].tolist())
#             prompt_ = (
#                 f"<|start_prompt|>Dataset description: {self.description}"
#                 f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
#                 "Input statistics: "
#                 f"min value {min_values_str}, "
#                 f"max value {max_values_str}, "
#                 f"median value {median_values_str}, "
#                 f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
#                 f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
#             )
# 
#             prompt.append(prompt_)
# 
#         # 鎭㈠鍘熷缁村害
#         x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
# 
#         # 灏嗘彁绀鸿浆鎹负token骞惰幏鍙栧祵鍏?#         prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
#         prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
# 
#         # 琛ヤ竵宓屽叆
#         x_enc = x_enc.permute(0, 2, 1).contiguous()
#         enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
# 
#         # 鐩存帴绾挎€ф槧灏勬浛浠atch-programming
#         enc_out = self.direct_mapping(enc_out)
# 
#         # 灏嗘彁绀哄祵鍏ュ拰杈撳叆鐗瑰緛宓屽叆杩炴帴璧锋潵
#         llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
# 
#         # 浣跨敤LLM澶勭悊杈撳叆
#         dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
#         dec_out = dec_out[:, :, :self.d_ff]
# 
#         dec_out = torch.reshape(
#             dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
#         dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
# 
#         dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
#         dec_out = dec_out.permute(0, 2, 1).contiguous()
# 
#         dec_out = self.normalize_layers(dec_out, 'denorm')
# 
#         return dec_out
# 
#     # 浣跨敤FFT锛堝揩閫熷倕閲屽彾鍙樻崲锛夎绠楄嚜鐩稿叧鍑芥暟锛岀劧鍚庢彁鍙杢op-k鐨勬粸鍚庡€?#     def calcute_lags(self, x_enc):
#         q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
#         k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
#         res = q_fft * torch.conj(k_fft)
#         corr = torch.fft.irfft(res, dim=-1)
#         mean_value = torch.mean(corr, dim=1)
#         _, lags = torch.topk(mean_value, self.top_k, dim=-1)
#         return lags
# 
# # 鍒犻櫎ReprogrammingLayer绫伙紝鍥犱负鍦ㄦ秷铻嶅疄楠屼腑涓嶉渶瑕
