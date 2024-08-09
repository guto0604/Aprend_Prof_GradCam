## Código equivalente ao apresentado em no Jupyter Notebook para o R. O uso das bibliotecas Keras e TensorFlow é bem parecido entre as linguagens.

## Mapas de Ativação Grad-Cam

library(keras)
library(reticulate)
library(imager)
library(magrittr)

# Carregar o modelo VGG16 pré-treinado
model <- application_vgg16(weights = "imagenet")

# Função para obter o mapa de calor Grad-CAM
get_gradcam_heatmap <- function(model, img_array, class_index) {
  # Obter a última camada convolucional
  last_conv_layer <- model$get_layer("block5_conv3")
  heatmap_model <- keras_model(inputs = model$input, outputs = list(last_conv_layer$output, model$output))
  
  with(tf$GradientTape() %as% tape, {
    conv_outputs, predictions <- heatmap_model(img_array)
    loss <- predictions[, class_index]
  })
  
  # Obter os gradientes da perda em relação às saídas da camada convolucional
  grads <- tape$gradient(loss, conv_outputs)
  pooled_grads <- tf$reduce_mean(grads, axis = c(1, 2))
  
  # Criar o mapa de ativação
  conv_outputs <- conv_outputs[1,,,]
  heatmap <- conv_outputs %*% pooled_grads
  heatmap <- tf$nn$relu(heatmap)
  heatmap <- heatmap / tf$reduce_max(heatmap)
  
  return(as.array(heatmap))
}

# Função para exibir o mapa de calor Grad-CAM
display_heatmap <- function(heatmap, img_path) {
  img <- load.image(img_path)
  
  plot(img, axes = FALSE)
  plot(heatmap, add = TRUE, col = terrain.colors(256), alpha = 0.5)
}

# Função para preprocessar a imagem
get_img_array <- function(img) {
  img_array <- image_to_array(img)
  img_array <- array_reshape(img_array, c(1, dim(img_array)))
  img_array <- imagenet_preprocess_input(img_array)
  return(img_array)
}

# Função para obter as principais previsões
get_top_predictions <- function(preds) {
  decoded <- imagenet_decode_predictions(preds, top = 5)[[1]]
  previsoes <- sapply(decoded, function(x) {
    sprintf("Raça: %-20s Prob: %.2f%%", x[[2]], x[[3]] * 100)
  })
  return(previsoes)
}

# Caminho para a imagem de exemplo
img_path <- "data/dogs/dog.10.jpg"

# Carregar e preprocessar a imagem
img <- image_load(img_path, target_size = c(224, 224))
img_array <- get_img_array(img)

# Fazer a previsão
preds <- model %>% predict(img_array)
class_index <- which.max(preds[1,])

# Gerar o mapa de calor Grad-CAM
heatmap <- get_gradcam_heatmap(model, img_array, class_index)

# Exibir a imagem e o mapa de calor
display_heatmap(heatmap, img_path)
print(get_top_predictions(preds))



## Lime


library(keras)
library(imager)
library(lime)
library(tensorflow)

# Carregar o modelo VGG16 pré-treinado
model <- application_vgg16(weights = "imagenet")
model %>% compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# Carregar e preprocessar a imagem
img_path <- "data/dogs/dog.10.jpg"
img <- load.image(img_path)
img <- resize(img, 224, 224)
img_array <- array_reshape(as.array(img), c(1, 224, 224, 3))
img_array <- imagenet_preprocess_input(img_array)

# Função para previsão
predict_proba <- function(x) {
  x <- imagenet_preprocess_input(x)
  predict(model, x)
}

# Criar o explicador LIME
explainer <- lime(image = img_array, model = predict_proba)

# Segmentar a imagem e gerar a explicação
explanation <- explain(image = img_array[1,,,], explainer = explainer, n_labels = 1, n_features = 5, segmentation_type = "slic")

# Extrair a imagem explicada e a máscara
temp <- explanation$superpixels
mask <- explanation$superpixel_selection

# Exibir a imagem com a explicação e a máscara
plot(temp)
plot(mask)

