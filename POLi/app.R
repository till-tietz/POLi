library(dplyr)
library(stringr)
library(magrittr)
library(plotly)
library(shiny)
library(tfdeploy)
library(keras)
library(shinyjs)
library(stopwords)
library(SnowballC)


label_mapping <- readRDS("www/label_mapping.rds")
tokenizer <- load_text_tokenizer("www/tokenizer")

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

analysis <- function(analysis_input, prog = NULL){

  if (is.function(prog)) {
    text <- "preparing model input"
    prog(detail = text)
  }

  text <- as.data.frame(unlist(strsplit(analysis_input, "\\."))) 
  text <- clean_text(data = text, col = 1)[[1]]
  
  sequences <- keras::texts_to_sequences(tokenizer, text)
  input_text <- keras::pad_sequences(sequences, maxlen = 45)
  input_text <- asplit(input_text,1)
  
  text <- as.data.frame(unlist(strsplit(analysis_input, "\\."))) 
  text <- clean_text(data = text, col = 1)[[1]]
  
  sequences <- keras::texts_to_sequences(tokenizer, text)
  input_text <- keras::pad_sequences(sequences, maxlen = 45)
  input_text <- asplit(input_text,1)
  
  if (is.function(prog)) {
    text <- "computing predictions"
    prog(detail = text)
  }
  
  results <- tfdeploy::predict_savedmodel(input_text, "www/saved_model/poli_model")$predictions %>%
    lapply(., function(i) t(as.vector(i[[1]]))) %>%
    do.call(rbind, .) %>%
    as.data.frame()
  
  colnames(results) <- as.character(label_mapping$label)
  results <- results %>%
    dplyr::summarise_all(., mean) %>%
    tidyr::pivot_longer(c(1:27)) %>%
    dplyr::rename(label = name,
                  probability = value) %>%
    dplyr::left_join(., label_mapping, by = "label") %>%
    dplyr::mutate(name = gsub("_", " ", label))
  
  
  plot <- ggplot(data = results, aes(x = reorder(name, probability), y = probability, label = description))+
    geom_bar(stat = "identity")+
    scale_y_continuous(limits = c(0,1), expand = c(0,0))+
    ylab("Classification Probability")+
    coord_flip()+
    theme_bw()+
    theme(axis.title.y = element_blank())
  return(plot)
}


ui <- navbarPage(
  "POLi (beta 0.1.0)",
  tabPanel(
    "Analysis",
    useShinyjs(),
    fluidRow(
      wellPanel(
        column(12, align = "center",
               textAreaInput(
                 "text_input",
                 label = NULL,
                 width = "4000px",
                 height = "100px",
                 placeholder = "enter your text...",
                 resize = "none"
               )
        ),
        actionButton("run","Analyze"),
        actionButton("clear","Clear")
      )
    ),
    fluidRow(
      column(12, align = "center",
             wellPanel(
               plotlyOutput("plotly_out", width = "75%") 
             )
  if (is.function(prog)) {
    text <- "computing predictions"
    prog(detail = text)
  }

  results <- tfdeploy::predict_savedmodel(input_text, "www/saved_model/poli_model")$predictions %>%
    lapply(., function(i) t(as.vector(i[[1]]))) %>%
    do.call(rbind, .) %>%
    as.data.frame()
  
  colnames(results) <- as.character(label_mapping$label)
  results <- results %>%
    dplyr::summarise_all(., mean) %>%
    tidyr::pivot_longer(c(1:27)) %>%
    dplyr::rename(label = name,
                  probability = value) %>%
    dplyr::left_join(., label_mapping, by = "label") %>%
    dplyr::mutate(name = gsub("_", " ", label))
  

  plot <- ggplot(data = results, aes(x = reorder(name, probability), y = probability, label = description))+
    geom_bar(stat = "identity")+
    scale_y_continuous(limits = c(0,1), expand = c(0,0))+
    ylab("Classification Probability")+
    coord_flip()+
    theme_bw()+
    theme(axis.title.y = element_blank())
  return(plot)
}


ui <- navbarPage(
  "POLi (beta 0.1.0)",
  tabPanel(
    "Analysis",
    useShinyjs(),
    fluidRow(
      wellPanel(
        column(12, align = "center",
               textAreaInput(
                 "text_input",
                 label = NULL,
                 width = "4000px",
                 height = "100px",
                 placeholder = "enter your text...",
                 resize = "none"
                 )
               ),
        actionButton("run","Analyze"),
        actionButton("clear","Clear")
      )
    ),
    fluidRow(
      column(12, align = "center",
            wellPanel(
              plotlyOutput("plotly_out", width = "75%") 
            )
          )
    )
  ),
  tabPanel(
    "About",
    fluidRow(
      wellPanel(
        "POLi is an experimental natural language processing model designed to predict the ideological position
             expressed in german political text. It utilizes custom word embeddings fed into a CNN
             neural network architecture to generate ideology predictions. POLi was trained on ideology
             labeled german party manifesto text data from The Manifesto Project ¹. As the data set used to
             train POLi is comparitively small and domain specific, the model will likely not generalize well beyond the immediate context of german political texts (i.e. parliamentary bills,
             policy proposals, parliamentary inquiries etc.). Within this domain; however, initial tests indicate promising performance."
      ),
      tags$footer(
        "¹ Volkens, Andrea / Burst, Tobias / Krause, Werner / Lehmann, Pola / Matthiess Theres / Merz, Nicolas / Regel, Sven / Wessels, Bernhard / Zehnter, Lisa (2020): The Manifesto Data Collection. Manifesto Project (MRG/CMP/MARPOR). Version 2020b. Berlin: Wissenschaftszentrum Berlin fuer Sozialforschung (WZB).",
        align = "left",
        style = "
              position:absolute;
              bottom:0;
              width:100%;
              height:50px;
              color: black;
              padding: 10px;
              z-index: 1000;"
      )
    )
  )
)


server <- function(input, output, session) {

  values <- reactive(input$text_input)

  plot <- eventReactive(input$run,{

    progress <- shiny::Progress$new()
    progress$set(message = "running model", value = 0)
    on.exit(progress$close())

    updateProgress <- function(value = NULL, detail = NULL) {
      if (is.null(value)) {
        value <- progress$getValue()
        value <- value + (progress$getMax() - value) / 2
      }
      progress$set(value = value, detail = detail)
    }

    analysis(analysis_input = values(), prog = updateProgress)
  })

  output$plotly_out <- renderPlotly(
    plotly_plot <- ggplotly(plot(), tooltip = c("probability","description"))%>%
      style(hoverlabel = list(align = "left"))
  )

  observeEvent(input$clear,{
    reset("text_input")
  })

  session$allowReconnect(FALSE)
}

shinyApp(ui, server)


