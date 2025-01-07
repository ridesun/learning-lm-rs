use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let vec_u8_to_vec_f32:fn(Vec<u8>) -> Vec<f32> =|bytes:Vec<u8>|{
            assert_eq!(bytes.len() % 4, 0);

            let mut f32_vec = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                f32_vec.push(value);
            }
            f32_vec
        };
        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let data = tensor.data().to_vec();
            let data_f32=vec_u8_to_vec_f32(data);
            Tensor::new(data_f32, &shape)
        };

        let embedding_table = get_tensor("lm_head.weight");

        let mut rms_att_w = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        for i in 0..config.num_hidden_layers {
            rms_att_w.push(get_tensor(&format!(
                "model.layers.{}.input_layernorm.weight",
                i
            )));
            wq.push(get_tensor(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                i
            )));

            wk.push(get_tensor(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                i
            )));
            wv.push(get_tensor(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                i
            )));
            wo.push(get_tensor(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                i
            )));
        }

        let mut rms_ffn_w = Vec::new();
        let mut w_up = Vec::new();
        let mut w_gate = Vec::new();
        let mut w_down = Vec::new();
        for i in 0..config.num_hidden_layers {
            rms_ffn_w.push(get_tensor(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                i
            )));
            w_up.push(get_tensor(&format!(
                "model.layers.{}.mlp.up_proj.weight",
                i
            )));
            w_gate.push(get_tensor(&format!(
                "model.layers.{}.mlp.gate_proj.weight",
                i
            )));

            w_down.push(get_tensor(&format!(
                "model.layers.{}.mlp.down_proj.weight",
                i
            )));
        }

        let rms_out_w = get_tensor("model.norm.weight");
        let lm_head = get_tensor("lm_head.weight");

        LLamaParams {
            embedding_table,

            rms_att_w,
            wq,
            wk,
            wv,
            wo,

            rms_ffn_w,
            w_up,
            w_gate,
            w_down,

            rms_out_w,
            lm_head,
        }
    }
}
