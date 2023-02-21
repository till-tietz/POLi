rm(list = ls())

if(!"pacman" %in% installed.packages()) {
  install.packages("pacman")
}

pacman::p_load(dplyr,tidytext,stopwords,SnowballC,
               text2vec,keras,data.table,magrittr)


## load data -------------------------------------------------------------------

# text data + ideology codes 
data <- list.files(path = "~/POLi/ideology_analysis_data/germany",
                   pattern = "\\.csv$",
                   full.names = TRUE) %>%
  purrr::map_dfr(data.table::fread, header = TRUE, fill = TRUE) %>%
  dplyr::select(text,cmp_code) %>%
  dplyr::rename(code = cmp_code) %>%
  tidyr::drop_na() 
 

# codes + labels  
codebook <- readxl::read_xlsx("~/POLi/codebook/codebook.xlsx") %>%
  dplyr::select(2:6) %>%
  dplyr::mutate(title = gsub(" ","-",gsub(":","",title)))


data <- dplyr::left_join(data, codebook, by = "code")

# clean text -------------------------------------------------------------------

# remove non alpha-numeric symbols + digits
# remove excess white space + stopwords 
# stem tokens
# this improves prediction accuracy 
clean_text <- function(data, cols) {
  
  for(i in 1:length(cols)) {
    data[[cols[i]]] <- data[[cols[i]]] %>%
      tolower() %>%
      stringr::str_replace_all("[^[:alnum:] ]+", " ") %>%
      stringr::str_replace_all("[[:digit:]]+", " ") %>%
      trimws(which = "both") %>%
      stringr::str_replace_all("[\\s]+", " ") %>%
      strsplit(.," ") %>%
      {
        stopwords <- stopwords::stopwords("de", source = "snowball")
        lapply(., function(i) paste(SnowballC::wordStem(i[!i %in% stopwords],"german"), collapse = " ")) %>%
          unlist()
      } 
  }
  
  return(data)
}

data <- clean_text(data,1) %>%
  dplyr::mutate(sentence_id = 1:dplyr::n()) %>%
  dplyr::select(sentence_id,text,title_aggregated) %>%
  magrittr::set_colnames(c("sentence_id","text","label")) 

rm(codebook)

## prep model data -------------------------------------------------------------
#change encoding
data$text <- iconv(data$text, "UTF-8", "latin1")

#drop sentences with fewer than 3 words >> necessary for creating skipgrams with window = 3 
drop <- strsplit(data$text," ") %>% 
  lapply(length) %>% 
  unlist() %>% 
  {. < 3} 

data <- data[!drop,]

# create label mapping >> allows us to join text labels to predictions later
data <- data %>%
  dplyr::mutate(numeric_label = as.numeric(as.factor(label)) - 1)

label_mapping <- data %>%
  dplyr::select(numeric_label, label) %>%
  dplyr::distinct(numeric_label, .keep_all = TRUE) %>%
  dplyr::arrange(numeric_label)


# create model data 
make_model_data <- function(data, x, y, n_classes, sets, probs, sampling_strat = NULL) {
  
  if(!all(sets %in% c("train","test","valid"))) {
    stop("sets must contain only 'train','test' and 'valid'")
  }
  
  if(length(sets) != length(probs)) {
    stop("must specify a sampling probability for each set")
  }
  
  if(!is.null(sampling_strat)) {
    if(!sampling_strat %in% c("up","down")) {
      stop("sampling_strat must be one of 'up' or 'down'")
    }
  }
  
  data <- as.data.frame(data)
  set <- sample(sets,nrow(data), prob = probs,replace = TRUE)
  
  #prep x
  x_val <- data[,x]
  num_words <- length(unique(unlist(strsplit(x_val," "))))
  
  #prep y
  y_val <- keras::to_categorical(as.matrix(data[,y]), num_classes = n_classes)
  
  #prep sequences
  tokenizer <- keras::text_tokenizer(num_words = num_words) %>% keras::fit_text_tokenizer(x_val)
  sequences <- keras::texts_to_sequences(tokenizer, x_val)
  max_length <- max(sapply(sequences, length))
  input <- keras::pad_sequences(sequences, maxlen = max_length)
  
  tokenizer_data <- list(tokenizer = tokenizer, max_length = max_length, num_words = num_words)
  model_data <- vector(mode = "list", length = length(sets))
  
  # make data for each set
  for(i in 1:length(sets)) {
    pos <- (set == sets[i])
    
    # apply imbalance fix to training data
    if(sets[i] == "train" & !is.null(sampling_strat)) {
      d <- data.frame(
        pos = which(pos),
        y = data[pos,y]
      )
      
      if(sampling_strat == "up") {
        d <- caret::upSample(x = d[,-ncol(d)],
                             y = factor(d$y))
      } else {
        d <- caret::downSample(x = d[,-ncol(d)],
                               y = factor(d$y))
      }
      
      pos <- d$x
    }
    
    model_data[[i]] <- list(
      x = x_val[pos],
      y = y_val[pos,],
      sequences = sequences[pos],
      input = input[pos,]
    )
  }
  
  names(model_data) <- sets
  model_data <- c(model_data,tokenizer_data)
  return(model_data)
  
}

model_data <- make_model_data(data = data,
                              x = "text",
                              y = "numeric_label",
                              n_classes = 27,
                              sets = c("train","test","valid"),
                              probs = c(0.5,0.3,0.2),
                              sampling_strat = "up")

## word2vec embedding ---------------------------------------------------------- 
# generate a 200D embedding using skipgrams 
skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
  
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  
  function() {
    skip <- generator_next(gen) %>%
      skipgrams(
        vocabulary_size = tokenizer$num_words, 
        window_size = window_size, 
        negative_samples = 1
      )
    
    x <- transpose(skip$couples) %>% purrr::map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    
    list(x, y)
  }
  
}

# making architecture
input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

embedding <- layer_embedding(
  input_dim = model_data$tokenizer$num_words + 1,
  output_dim = 200,
  input_length = 1, 
  name = "embedding"
)

target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten() 

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)

output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

# fit model 
model %>%
  keras::fit(
    skipgrams_generator(as.vector(data$text), model_data$tokenizer, 3, 2),
    steps_per_epoch = 1000, epochs = 5
  )


# get embedding weights
word2vec_embedding <- keras::get_weights(model)[[1]]
rownames(word2vec_embedding) <- c('NA',as.vector(unlist(model_data$tokenizer$index_word)))

#saveRDS(word2vec_embedding, "word2vec_embedding_agglabel.rds")

# check embedding
cos_sim <- function(token, n_matches) {
  embedding_vector <- t(matrix(word2vec_embedding[token,])) 
  cos_sim = sim2(x = word2vec_embedding, y = embedding_vector, method = "cosine", norm = "l2")
  print(head(sort(cos_sim[,1], decreasing = TRUE), n_matches))
}
cos_sim("bildung",10)

rm(context_vector,dot_product,input_context,input_target,output,target_vector,
   embedding,model,skipgrams_generator)
gc()

## build model -----------------------------------------------------------------

#word2vec_embedding <- readRDS("word2vec_embedding_agglabel.rds")
#m <- keras::keras_model_sequential() %>% 
#  keras::layer_embedding(input_dim = model_data$num_words + 1,
#                         output_dim = ncol(word2vec_embedding), 
#                         input_length = model_data$max_length,
#                         mask_zero = TRUE,   
#                         weights = list(word2vec_embedding), 
#                         trainable = FALSE)%>% 
#  keras::layer_conv_1d(filters = 300, kernel_size = 5) %>%
#  keras::layer_spatial_dropout_1d(rate = 0.2) %>%
#  keras::bidirectional(keras::layer_lstm(units = 300, 
#                                         return_sequences = TRUE, 
#                                         dropout = 0.2, 
#                                         recurrent_dropout = 0.02, 
#                                         activity_regularizer = keras::regularizer_l2(l = 0.05))) %>%
#  keras::layer_flatten() %>%
#  keras::layer_dense(units = 27, activation = "softmax") 


m <- keras::keras_model_sequential() %>% 
  keras::layer_embedding(input_dim = model_data$num_words + 1,
                         output_dim = ncol(word2vec_embedding), 
                         input_length = model_data$max_length,
                         mask_zero = TRUE,   
                         weights = list(word2vec_embedding), 
                         trainable = FALSE) %>%
  keras::layer_conv_1d(filters = 200, kernel_size = 5, activation = 'relu') %>%
  keras::layer_max_pooling_1d(pool_size = 5) %>%
  keras::layer_conv_1d(filters = 200, kernel_size = 5, activation = 'relu') %>%
  keras::layer_global_max_pooling_1d() %>%
  keras::layer_dense(units = 27, activation = "softmax")


m %>% 
  keras::compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("categorical_accuracy")
  )


history <- m %>% 
  keras::fit(
    model_data$train$input, 
    model_data$train$y,
    epochs = 5,
    batch_size = 2000,
    validation_data = list(model_data$valid$input, model_data$valid$y)
  )


## model evaluation ------------------------------------------------------------
get_pred <- function(model_data, model, n, label_mapping, new_data = NULL) {
  
  if(!is.null(new_data)) {
    input <- clean_text(new_data,"text") %>%
      dplyr::pull(text) %>%
      keras::texts_to_sequences(model_data$tokenizer, .) %>%
      keras::pad_sequences(., maxlen = model_data$max_length)
    
    data_pred <- as.data.frame(predict(model, input)) %>%
      apply(., 1, doBy::which.maxn, n)
  } else {
    data_pred <- as.data.frame(predict(model, model_data$test$input)) %>%
      apply(., 1, doBy::which.maxn, n)  
  }
  
  if(n > 1) {
    data_pred <- data_pred %>%
      t() %>%
      as.data.frame() %>%
      dplyr::mutate(id = 1:dplyr::n()) %>%
      tidyr::pivot_longer(!id, names_to = "rank", values_to = "numeric_label")%>%
      dplyr::mutate(rank = as.numeric(gsub("V","",rank)),
                    numeric_label = numeric_label - 1) 
  } else {
    data_pred <- data_pred %>%
      {data.frame(numeric_label = .)} %>%
      dplyr::mutate(id = 1:dplyr::n(),
                    numeric_label = numeric_label - 1)
  }
  
  data_pred <- data_pred %>% 
    dplyr::left_join(., label_mapping, by = "numeric_label")%>%
    dplyr::rename(numeric_label_pred = numeric_label,
                  label_pred = label)
    
  
  if(is.null(new_data)) {
    data_true <- data.frame(
      numeric_label = apply(model_data$test$y,1, which.max) - 1
    ) %>%
      dplyr::mutate(id = 1:dplyr::n()) %>%
      dplyr::left_join(., label_mapping, by = "numeric_label")
    
    data_pred <- dplyr::left_join(data_pred, data_true, by = "id")
  }
  
  return(data_pred)
}

results <- get_pred(model_data, m, 3, label_mapping)

results %>%
  dplyr::group_by(id) %>%
  dplyr::mutate(correct = numeric_label %in% numeric_label_pred) %>%
  dplyr::pull(correct) %>%
  {sum(.) / length(.)}

## test new sentences ----------------------------------------------------------

new_sentences <- data.frame(text = c("Wir fordern eine strikte Einwanderungspolitik.",
                                     "Der Ausbau erneuerbarer Energien ist essentiell in der Bekämpfung des Klimawandels",
                                     "Ausbau des Sozialstaates ist negativ behaftet.",
                                     "Auftrag der Mission Resolute Support bleibt es, die nationalen Verteidigungs und  Sicherheitskräfte  zu  befähigen,  ihrer  Sicherheitsverantwortung  nachzukommen.",
                                     "Strom aus erneuerbaren Energien leistet einen wesentlichen Beitrag zu Erreichung der  Klimaziele  Deutschlands  und  der  Europäischen  Union.",
                                     "Es ist eine Kernaufgabe staatlichen Handelns, Rahmenbedingungen für gesunde, sichere und menschengerecht gestaltete Arbeitsbedingungen der Beschäftigten zu schaffen.",
                                     "Die Erreichung der globalen Nachhaltigkeitsziele und der Pariser Klimaschutz-ziele darf auch im weltweiten Kampf gegen die Folgen der COVID-19-Pandemie nicht  in  den  Hintergrund  der  internationalen  Zusammenarbeit  rücken,  sondern  muss   weiterhin zentral für das Handeln der internationalen Staatengemeinschaft bleiben.",
                                     "Die Realisierung von Nord Stream 2 trägt zu sinkenden Gaspreisen bei, da auf Grund der im Vergleich zur Route über Festland kürzeren Transportroute durch die Ostsee Transportkosten sowie Transitgebühren eingespart werden.",
                                     "Die kürzere Transportroute und der Umstand, dass es sich um eine  moderne Pipeline handelt, verringern zugleich die allgemeine Umweltbelastung und das Risiko eines Schadensfalles durch Abnutzung oder veralteter Technik samt hieraus entstehender Umweltbelastungen."))

get_pred(model_data, m, 1, label_mapping, new_sentences) %>%
  dplyr::pull(label_pred)











