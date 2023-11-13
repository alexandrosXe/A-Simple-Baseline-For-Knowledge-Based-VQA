
import numpy as np 
import pandas as pd
from torch.nn.functional import cosine_similarity

def sort_captions_based_on_similarity(captions,raw_image,model,processor, device = "cuda", ascending = False):
  """
  Rank the qr captions based on their similarity with the image
  :param captions: The captions that will be ranked 
  :param raw_image: The PIL image object 
  :param model: The image-to-text similarity model (BLIP)
  :param processor: The image and text processor 
  :param device: Cpu or Gpu
  :param ascending: Bool variable for ranking the captions at ascending order or not 
  :returns results_df: Captions ranked 
  :returns cosine_scores: The cosine score of each caption with the image
  """
  #encode the captions
  text_input = processor(text = captions, return_tensors="pt", padding = True).to(device)
  text_embeds = model.text_encoder(**text_input)
  text_embeds = text_embeds[0]
  text_features = model.text_proj(text_embeds[:, 0, :])

  #encode the image 
  image_input = processor(images=raw_image, return_tensors="pt").to(device)
  vision_outputs = model.vision_model(**image_input)
  image_embeds = vision_outputs[0]
  image_feat = model.vision_proj(image_embeds[:, 0, :])
  
  #compute cos sim
  cosine_scores = cosine_similarity(text_features, image_feat).tolist()

  #sort captions based on the cosine scores
  captions = [x for _, x in sorted(zip(cosine_scores, captions), reverse = True)]
  cosine_scores.sort(reverse = True)
  return captions, cosine_scores

def get_context_examples(sample_q_embed, sample_i_embed, train_q_embedds, train_i_embedds, n_shots):
  """
  Get the n context examples for n-shot in context learning
  according to the avg img and question similarities

  :param sample_q_embed: The normalized question embedding of the test sample (shot)
  :param sample_i_embed: The normalized image embedding of the test sample (shot)
  :param train_q_embedds: Dataframe containing the normalized question embeddings of the train samples (shots)
  :param train_i_embedds: Dataframe containing the normalized image embeddings of the train samples (shots)
  :param n_shots: The number of training examples (shots) to return 
  :returns results_df: Dataframe containing the n_shot most similar examples to the test sample


  """
  #compute question sims 
  q_sims_df = train_q_embedds.copy(deep = True)
  q_sims_df['q_cos_sim'] = q_sims_df.question_embedd.apply(lambda x: np.matmul(x, sample_q_embed))

  #compute image sims 
  i_sims_df = train_q_embedds.copy(deep = True)
  i_sims_df['i_cos_sim'] = i_sims_df.question_embedd.apply(lambda x: np.matmul(x, sample_i_embed))

  results_df = pd.merge(q_sims_df,i_sims_df,on='question_id')
  
  results_df['avg_cos_sim'] = results_df.apply(lambda row: (row["q_cos_sim"] + row["i_cos_sim"])/2, axis = 1)
  results_df = results_df.sort_values(by = 'avg_cos_sim', ascending = False)
  return results_df[:n_shots]
