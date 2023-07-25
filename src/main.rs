use std::{env, io, mem, slice};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::process::exit;
use std::time::Instant;
use rand::prelude::ThreadRng;
use rand::Rng;

#[derive(Debug, Default)]
struct Config {
    // transformer dimension
    pub dim: i32,
    // for ffn layers
    pub hidden_dim: i32,
    // number of layers
    pub n_layers: i32,
    // number of query heads
    pub n_heads: i32,
    // number of key/value heads (can be < query heads because of multiquery)
    #[allow(dead_code)]
    pub n_kv_heads: i32,
    // vocabulary size, usually 256 (byte-level)
    pub vocab_size: i32,
    // max sequence length
    pub seq_len: i32,
}

#[derive(Debug, Default)]
struct TransformerWeights {
    // token embedding table
    // (vocab_size, dim)
    pub token_embedding_table: Vec<f32>,
    // (layer, dim) rmsnorm weights
    pub rms_att_weight: Vec<f32>,
    // (layer, dim)
    pub rms_ffn_weight: Vec<f32>,
    // (layer, dim, dim)
    pub wq: Vec<f32>,
    // (layer, dim, dim)
    pub wk: Vec<f32>,
    // (layer, dim, dim)
    pub wv: Vec<f32>,
    // (layer, dim, dim)
    pub wo: Vec<f32>,
    // (layer, hidden_dim, dim)
    pub w1: Vec<f32>,
    // (layer, dim, hidden_dim)
    pub w2: Vec<f32>,
    // (layer, hidden_dim, dim)
    pub w3: Vec<f32>,
    // (dim,)
    pub rms_final_weight: Vec<f32>,
    // (seq_len, dim/2)
    pub freq_cis_real: Vec<f32>,
    // (seq_len, dim/2)
    pub freq_cis_imag: Vec<f32>,
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Vec<f32>,
}

struct RunState {
    pub x: Vec<f32>,
    // activation at current time stamp (dim,)
    pub xb: Vec<f32>,
    // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,
    // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,
    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,
    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,
    // query (dim,)
    pub k: Vec<f32>,
    // key (dim,)
    pub v: Vec<f32>,
    // value (dim,)
    pub att: Vec<f32>,
    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>,
    // output logits
    pub key_cache: Vec<f32>,
    // (layer, seq_len, dim)
    pub value_cache: Vec<f32>, // (layer, seq_len, dim)
}

fn new_run_state(c: &Config) -> RunState {
    RunState {
        x: vec![0.0f32; c.dim as usize],
        xb: vec![0.0f32; c.dim as usize],
        xb2: vec![0.0f32; c.dim as usize],
        hb: vec![0.0f32; c.hidden_dim as usize],
        hb2: vec![0.0f32; c.hidden_dim as usize],
        q: vec![0.0f32; c.dim as usize],
        k: vec![0.0f32; c.dim as usize],
        v: vec![0.0f32; c.dim as usize],
        att: vec![0.0f32; (c.n_heads * c.seq_len) as usize],
        logits: vec![0.0f32; c.vocab_size as usize],
        key_cache: vec![0.0f32; (c.n_layers * c.seq_len * c.dim) as usize],
        value_cache: vec![0.0f32; (c.n_layers * c.seq_len * c.dim) as usize],
    }
}

fn try_new_transformer_weights(config: &Config, file: &mut File, shared_weights: bool) -> Option<TransformerWeights> {
    let mut weights = TransformerWeights {
        token_embedding_table: vec![0.0; (config.vocab_size * config.dim) as usize],
        rms_att_weight: vec![0.0; (config.n_layers * config.dim) as usize],
        rms_ffn_weight: vec![0.0; (config.n_layers * config.dim) as usize],
        wq: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
        wk: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
        wv: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
        wo: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
        w1: vec![0.0; (config.n_layers * config.hidden_dim * config.dim) as usize],
        w2: vec![0.0; (config.n_layers * config.dim * config.hidden_dim) as usize],
        w3: vec![0.0; (config.n_layers * config.hidden_dim * config.dim) as usize],
        rms_final_weight: vec![0.0; config.dim as usize],
        freq_cis_real: vec![0.0; (config.seq_len * (config.dim / config.n_heads) / 2) as usize],
        freq_cis_imag: vec![0.0; (config.seq_len * (config.dim / config.n_heads) / 2) as usize],
        wcls: vec![0.0; (config.vocab_size * config.dim / 2) as usize],
    };

    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.token_embedding_table.as_mut_ptr() as *mut u8,
            weights.token_embedding_table.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.rms_att_weight.as_mut_ptr() as *mut u8,
            weights.rms_att_weight.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.wq.as_mut_ptr() as *mut u8,
            weights.wq.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.wk.as_mut_ptr() as *mut u8,
            weights.wk.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.wv.as_mut_ptr() as *mut u8,
            weights.wv.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.wo.as_mut_ptr() as *mut u8,
            weights.wo.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.rms_ffn_weight.as_mut_ptr() as *mut u8,
            weights.rms_ffn_weight.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.w1.as_mut_ptr() as *mut u8,
            weights.w1.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.w2.as_mut_ptr() as *mut u8,
            weights.w2.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.w3.as_mut_ptr() as *mut u8,
            weights.w3.len() * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.rms_final_weight.as_mut_ptr() as *mut u8,
            weights.rms_final_weight.len() * mem::size_of::<f32>(),
        )
    }).unwrap();

    let head_size = (config.dim / config.n_heads) as usize;
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.freq_cis_real.as_mut_ptr() as *mut u8,
            config.seq_len as usize * head_size / 2 * mem::size_of::<f32>(),
        )
    }).unwrap();
    file.read_exact(unsafe {
        slice::from_raw_parts_mut(
            weights.freq_cis_imag.as_mut_ptr() as *mut u8,
            config.seq_len as usize * head_size / 2 * mem::size_of::<f32>(),
        )
    }).unwrap();

    if shared_weights {
        weights.wcls = weights.token_embedding_table.clone();
    } else {
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wcls.as_mut_ptr() as *mut u8,
                weights.wcls.len() * mem::size_of::<f32>(),
            )
        }).unwrap();
    }

    Some(weights)
}

fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    let mut ss = 0.0f32;
    for j in 0..size {
        ss += x[j] * x[j];
    }
    ss /= size as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn rmsnorm2(o: &mut [f32], weight: &[f32], size: usize) {
    let mut ss: f32 = 0.0;
    for j in 0..size {
        ss += o[j] * o[j];
    }
    ss /= size as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    for j in 0..size {
        o[j] = weight[j] * (ss * o[j]);
    }
}

fn softmax(x: &mut [f32], size: usize) {
    // find max value (for numerical stability)
    let mut max_val = 0f32;
    for i in 0..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    //exp and sum
    let mut sum = 0.0f32;
    for i in 0..size {
        x[i] = (x[i] - max_val).exp();
        sum += x[i]
    }
    // normalize
    for i in 0..size {
        x[i] /= sum;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..d {
        let mut val = 0.0f32;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn transformer(token: usize, pos: usize, p: &Config, s: &mut RunState, w: &TransformerWeights) {
    // a few convenience variables
    let x = &mut s.x;
    let dim = p.dim as usize;
    let hidden_dim = p.hidden_dim as usize;
    let head_size = dim / p.n_heads as usize;


    // copy the token embedding into x
    let content_row = &w.token_embedding_table[token * dim..token * dim + dim];
    x.copy_from_slice(content_row);

    // pluck out the "pos" row of freq_cis_real and rewq_cis_imag
    let freq_cis_real_row = &w.freq_cis_real[pos * head_size / 2..];
    let freq_cis_imag_row = &w.freq_cis_imag[pos * head_size / 2..];

    // forward all the layers
    for l in 0..p.n_layers as usize {
        // attention rmsnorm
        rmsnorm(&mut s.xb, x, &w.rms_att_weight[l * dim..], dim);

        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l * dim * dim..], dim, dim);
        matmul(&mut s.v, &s.xb, &w.wv[l * dim * dim..], dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..p.n_heads as usize {
            let q = &mut s.q[h * head_size..];
            let k = &mut s.k[h * head_size..];
            for i in (0..head_size).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        let loff = l * p.seq_len as usize * dim; // kv cache layer offset for convenience
        let key_cache_row = &mut s.key_cache[loff + pos * dim..loff + pos * dim + dim];
        let value_cache_row = &mut s.value_cache[loff + pos * dim..loff + pos * dim + dim];
        key_cache_row.copy_from_slice(&s.k[0..dim]);
        value_cache_row.copy_from_slice(&s.v[0..dim]);

        // multihead attention. iterate over all heads
        for h in 0..p.n_heads as usize {
            // get the query vector for this head
            let q = &s.q[h * head_size..];
            // attention scores for this head
            let att = &mut s.att[h * p.seq_len as usize..];
            for t in 0..=pos {
                let k = &s.key_cache[loff + t * dim + h * head_size..];
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values,,,,,, store back into xb
            for i in 0..head_size {
                let mut val = 0.0f32;
                for t in 0..=pos {
                    val += att[t] * s.value_cache[loff + t * dim + h * head_size + i]; // note bad locality
                }
                s.xb[h * head_size + i] = val;
            }
        }

        // final matmul to get the output fo the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..], dim, dim);


        // residual connection
        accum(x, &s.xb2, dim);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, &x, &w.rms_ffn_weight[l*dim..], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(&mut s.hb, &s.xb, &w.w1[l * dim * hidden_dim..], dim, hidden_dim);
        matmul(&mut s.hb2, &s.xb, &w.w3[l * dim * hidden_dim..], dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            s.hb[i] = s.hb[i] * (1.0f32 / (1.0f32 + (-s.hb[i]).exp()));
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(&mut s.xb, &s.hb, &w.w2[l * dim * hidden_dim..], hidden_dim, dim);

        // residual connection
        accum(x, &s.xb, dim);
    }

    //final rmsnorm
    rmsnorm2(x, &w.rms_final_weight, dim);

    // classifier into logits
    matmul(&mut s.logits, &x, &w.wcls, p.dim as usize, p.vocab_size as usize);
}

fn sample(probabilities: &[f32], n: usize, rng: &mut ThreadRng) -> usize {
    // sample index from probabilities, they must sum to 1
    let r: f32 = rng.gen();
    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if r < cdf {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

fn argmax(v: &[f32], n: usize) -> usize {
    let mut max_i = 0;
    let mut max_p = v[0];
    for i in 1..n {
        if v[i] > max_p {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

fn load_vocab(config: &Config) -> Option<Vec<String>> {
    let mut vocab: Vec<String> = Vec::with_capacity(config.vocab_size as usize);
    let file = File::open("tokenizer.bin").ok()?;
    let mut reader = BufReader::new(file);

    for _ in 0..config.vocab_size {
        let mut len_buf = [0u8; mem::size_of::<i32>()];
        reader.read_exact(&mut len_buf).ok()?;
        let len = i32::from_le_bytes(len_buf);

        let mut word_buf = vec![0u8; len as usize];
        reader.read_exact(&mut word_buf).ok()?;

        let word = String::from_utf8(word_buf).ok()?;
        vocab.push(word);
    }

    Some(vocab)
}


fn main() {
    let mut checkpoint = String::new(); // e.g. out/model.bin
    let mut temperature = 0.9f32; //e.g. 1.0, or 0.0
    let mut steps = 256; // max number of steps to run for, 0: use seq_len

    // 'checkpoint' is necessary arg
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print!("Usage: {} <checkpoint_file> [temperature] [steps]\n", args[0]);
        exit(1);
    }
    if args.len() >= 2 {
        checkpoint = args[1].clone();
    }
    if args.len() >= 3 {
        temperature = args[2].parse::<f32>().expect("wrong temperature");
    }
    if args.len() >= 4 {
        steps = args[3].parse::<usize>().expect("wrong steps");
    }

    let mut rng = rand::thread_rng();

    #[allow(unused_assignments)]
        let mut config = Config::default();

    let mut file = File::open(checkpoint.clone()).expect("fail to open checkpoint file");
    let mut buffer = [0u8; std::mem::size_of::<Config>()];

    // read in the config header
    file.read_exact(&mut buffer).expect("fail to read checkpoint file");
    config = unsafe { std::mem::transmute(buffer) };
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = config.vocab_size > 0;
    config.vocab_size = config.vocab_size.abs();

    let weights = try_new_transformer_weights(&config, &mut file, shared_weights).expect("fail to load weights");

    if steps <= 0 || steps > config.seq_len as usize { steps = config.seq_len as usize; }

    let vocab = load_vocab(&config).expect("fail to load vocab");

    let start_time = Instant::now();
    let mut next;
    let mut token = 1;
    let mut pos = 0;

    let mut state = new_run_state(&config);
    println!("<s>");
    while pos < steps {
        transformer(token, pos, &config, &mut state, &weights);

        if temperature == 0.0f32 {
            next = argmax(&state.logits, config.vocab_size as usize);
        } else {
            for q in 0..config.vocab_size {
                state.logits[q as usize] /= temperature;
            }
            softmax(&mut state.logits, config.vocab_size as usize);
            next = sample(&state.logits, config.vocab_size as usize, &mut rng);
        }
        let s = vocab[next].clone();
        print!("{}", s);
        let _ = io::stdout().flush();

        // advance forward
        token = next;
        pos += 1;
    }

    print!("\nachieved tok/s: {}\n", config.seq_len as u64 / start_time.elapsed().as_secs());
}
