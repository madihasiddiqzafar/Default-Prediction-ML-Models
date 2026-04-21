# =============================================================================
# ASSIGNMENT 2 — SHINY DASHBOARD
# Big Data Module - Bankruptcy Prediction
# University of Birmingham Business School
# =============================================================================
# Run this script in RStudio. It will launch an interactive dashboard
# in your browser that you can screen-record for your 15-minute video.
#
# BEFORE RUNNING: Make sure all .rds and .png files from Assignment 1
# scripts are in the same folder as this script.
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("shiny", "shinydashboard", "dplyr", "ggplot2",
              "pROC", "randomForest", "DT", "tidyr", "png", "grid")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(shiny)
library(shinydashboard)
library(dplyr)
library(ggplot2)
library(pROC)
library(randomForest)
library(DT)
library(tidyr)
library(png)
library(grid)

# =============================================================================
# LOAD DATA
# =============================================================================

lr_results  <- readRDS("lr_results.rds")
rf_results  <- readRDS("rf_results.rds")
nn_results  <- readRDS("nn_results.rds")
df_final    <- readRDS("df_final.rds")
train       <- readRDS("train_scaled.rds")
test        <- readRDS("test_scaled.rds")

# Metrics comparison table
comparison_df <- bind_rows(
  lr_results$metrics,
  rf_results$metrics,
  nn_results$metrics
)

# Confusion matrix summary
extract_cm <- function(results, model_name) {
  cm_table <- results$cm$table
  data.frame(
    Model          = model_name,
    True_Negative  = cm_table["No",  "No"],
    False_Positive = cm_table["Yes", "No"],
    False_Negative = cm_table["No",  "Yes"],
    True_Positive  = cm_table["Yes", "Yes"]
  )
}

cm_summary <- bind_rows(
  extract_cm(lr_results, "Logistic Regression"),
  extract_cm(rf_results, "Random Forest"),
  extract_cm(nn_results, "Neural Network")
)

# Feature importance from RF final model
importance_df <- data.frame(
  Feature          = rownames(importance(rf_results$model)),
  MeanDecreaseGini = importance(rf_results$model)[, "MeanDecreaseGini"],
  row.names        = NULL
) %>% arrange(desc(MeanDecreaseGini))

# =============================================================================
# UI
# =============================================================================

ui <- dashboardPage(
  skin = "black",

  dashboardHeader(
    title = span(
      style = "font-family: 'Georgia', serif; font-size: 16px; color: #F0C040;",
      "Bankruptcy Prediction — Big Data"
    ),
    titleWidth = 320
  ),

  dashboardSidebar(
    width = 260,
    tags$div(
      style = "padding: 15px 10px 5px 15px;",
      tags$p(style = "color: #AAA; font-size: 11px; margin: 0;",
             "University of Birmingham"),
      tags$p(style = "color: #AAA; font-size: 11px; margin: 0;",
             "Big Data — Assignment 2"),
      tags$hr(style = "border-color: #444; margin: 10px 0;")
    ),
    sidebarMenu(
      menuItem("1. Background",        tabName = "background",
               icon = icon("book-open")),
      menuItem("2. Steps Performed",   tabName = "steps",
               icon = icon("list-ol")),
      menuItem("3. How We Did It",     tabName = "howwedidit",
               icon = icon("code")),
      menuItem("4. Pros & Cons",       tabName = "proscons",
               icon = icon("balance-scale")),
      menuItem("5. Results",           tabName = "results",
               icon = icon("chart-bar")),
      menuItem("6. Model Comparison",  tabName = "comparison",
               icon = icon("trophy"))
    )
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      body, .content-wrapper, .main-sidebar, .sidebar {
        font-family: 'Georgia', serif;
        background-color: #1A1A2E;
      }
      .content-wrapper { background-color: #1A1A2E; }
      .box {
        background-color: #16213E;
        border-top: 3px solid #F0C040;
        border-radius: 6px;
        color: #E0E0E0;
      }
      .box-header { color: #F0C040 !important; }
      .box-title  { color: #F0C040 !important; font-size: 15px; }
      h2 { color: #F0C040; font-family: 'Georgia', serif; }
      h3 { color: #E0E0E0; font-family: 'Georgia', serif; }
      h4 { color: #F0C040; }
      p, li { color: #C8C8C8; line-height: 1.7; }
      .metric-box {
        background: #0F3460;
        border-left: 4px solid #F0C040;
        border-radius: 5px;
        padding: 15px 20px;
        margin: 8px 0;
      }
      .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #F0C040;
      }
      .metric-label {
        font-size: 12px;
        color: #AAA;
        text-transform: uppercase;
        letter-spacing: 1px;
      }
      .model-card {
        background: #0F3460;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid;
      }
      .card-lr { border-color: #2980B9; }
      .card-rf { border-color: #27AE60; }
      .card-nn { border-color: #8E44AD; }
      .step-card {
        background: #0F3460;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 8px 0;
        border-left: 4px solid #F0C040;
      }
      .badge-blue   { background:#2980B9; color:#fff;
                      padding:3px 10px; border-radius:12px;
                      font-size:11px; }
      .badge-green  { background:#27AE60; color:#fff;
                      padding:3px 10px; border-radius:12px;
                      font-size:11px; }
      .badge-purple { background:#8E44AD; color:#fff;
                      padding:3px 10px; border-radius:12px;
                      font-size:11px; }
      .badge-gold   { background:#F0C040; color:#000;
                      padding:3px 10px; border-radius:12px;
                      font-size:11px; }
      table.dataTable { color: #E0E0E0 !important; }
      .dataTables_wrapper { color: #E0E0E0; }
      .main-header .logo { background-color: #0D0D1A !important; }
      .main-header .navbar { background-color: #0D0D1A !important; }
      .main-sidebar { background-color: #0D0D1A !important; }
      .sidebar-menu > li > a { color: #CCC !important; }
      .sidebar-menu > li.active > a { color: #F0C040 !important;
                                      border-left: 3px solid #F0C040; }
    "))),

    tabItems(

      # -----------------------------------------------------------------------
      # TAB 1: BACKGROUND
      # -----------------------------------------------------------------------
      tabItem(tabName = "background",
        fluidRow(
          column(12,
            tags$h2("1. Background of Methods Used"),
            tags$p("This project applies machine learning and deep learning techniques
                    to predict corporate bankruptcy using financial ratio data from
                    6,819 Taiwanese companies.")
          )
        ),
        fluidRow(
          column(4,
            tags$div(class = "model-card card-lr",
              tags$h4(tags$span(class="badge-blue", "Model 1"),
                      " Logistic Regression"),
              tags$p("A statistical method that models the probability of a binary
                      outcome as a function of predictor variables. It is the
                      standard baseline in credit risk modelling (Altman & Sabato, 2007)
                      and produces fully interpretable coefficients."),
              tags$p(tags$strong(style="color:#2980B9", "Type:"),
                     tags$span(style="color:#CCC",
                               " Supervised | Linear | Parametric"))
            )
          ),
          column(4,
            tags$div(class = "model-card card-rf",
              tags$h4(tags$span(class="badge-green", "Model 2"),
                      " Random Forest"),
              tags$p("An ensemble method that trains hundreds of decision trees on
                      random subsets of data and features, then combines their
                      predictions. Robust to outliers and captures non-linear
                      relationships (Breiman, 2001)."),
              tags$p(tags$strong(style="color:#27AE60", "Type:"),
                     tags$span(style="color:#CCC",
                               " Supervised | Non-linear | Ensemble"))
            )
          ),
          column(4,
            tags$div(class = "model-card card-nn",
              tags$h4(tags$span(class="badge-purple", "Model 3"),
                      " Neural Network"),
              tags$p("A deep learning model with two hidden layers (16 and 8 nodes)
                      that learns complex non-linear representations through
                      backpropagation. Satisfies the deep learning learning
                      outcome (Tam & Kiang, 1992)."),
              tags$p(tags$strong(style="color:#8E44AD", "Type:"),
                     tags$span(style="color:#CCC",
                               " Supervised | Deep Learning | 2 Hidden Layers"))
            )
          )
        ),
        fluidRow(
          box(width = 12, title = "Dataset Overview",
            fluidRow(
              column(3, tags$div(class="metric-box",
                tags$div(class="metric-value", "6,819"),
                tags$div(class="metric-label", "Companies"))),
              column(3, tags$div(class="metric-box",
                tags$div(class="metric-value", "95"),
                tags$div(class="metric-label", "Original Features"))),
              column(3, tags$div(class="metric-box",
                tags$div(class="metric-value", "22"),
                tags$div(class="metric-label", "Selected Features"))),
              column(3, tags$div(class="metric-box",
                tags$div(class="metric-value", "3.2%"),
                tags$div(class="metric-label", "Bankruptcy Rate")))
            )
          )
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 2: STEPS PERFORMED
      # -----------------------------------------------------------------------
      tabItem(tabName = "steps",
        fluidRow(
          column(12, tags$h2("2. Steps Performed"))
        ),
        fluidRow(
          column(6,
            tags$div(class="step-card",
              tags$h4("Step 1 — Data Quality Control"),
              tags$ul(
                tags$li("Assessed dataset size: 6,819 rows × 96 columns"),
                tags$li("Confirmed zero missing values across all features"),
                tags$li("Identified severe class imbalance (96.8% vs 3.2%)"),
                tags$li("Detected outliers using the IQR method"),
                tags$li("Visualised feature distributions and correlations")
              )
            ),
            tags$div(class="step-card",
              tags$h4("Step 2 — Feature Selection"),
              tags$ul(
                tags$li("Removed near-zero variance features"),
                tags$li("Ranked remaining features by Random Forest Gini importance"),
                tags$li("Selected top 20 features from importance ranking"),
                tags$li("Force-included 2 binary distress flags on theoretical grounds
                         (Altman, 1968; Zmijewski, 1984)"),
                tags$li("Final feature set: 22 variables")
              )
            ),
            tags$div(class="step-card",
              tags$h4("Step 3 — Data Preparation"),
              tags$ul(
                tags$li("Stratified 80/20 train/test split"),
                tags$li("Applied SMOTE to training set only (10,560 balanced rows)"),
                tags$li("Standardised continuous features (mean=0, SD=1)"),
                tags$li("Binary flags left unscaled to preserve meaning")
              )
            )
          ),
          column(6,
            tags$div(class="step-card",
              tags$h4("Step 4 — Model Development"),
              tags$ul(
                tags$li("Trained Logistic Regression with 10-fold cross-validation"),
                tags$li("Tuned Random Forest mtry via 5-fold CV (best mtry = 4)"),
                tags$li("Trained Neural Network: 2 hidden layers (16 → 8 nodes)"),
                tags$li("All models evaluated on the same held-out test set")
              )
            ),
            tags$div(class="step-card",
              tags$h4("Step 5 — Evaluation & Comparison"),
              tags$ul(
                tags$li("Computed confusion matrix for each model"),
                tags$li("Calculated Accuracy, Precision, Recall, F1, Specificity"),
                tags$li("Generated AUC-ROC scores and curves"),
                tags$li("Produced combined ROC plot for direct comparison"),
                tags$li("Identified best model per metric and overall")
              )
            ),
            tags$div(class="step-card",
              tags$h4("Step 6 — Interpretation"),
              tags$ul(
                tags$li("Interpreted feature importance in financial context"),
                tags$li("Evaluated models against real-world decision use cases"),
                tags$li("Discussed ethical considerations and GDPR implications"),
                tags$li("Acknowledged limitations of methodology")
              )
            )
          )
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 3: HOW WE DID IT
      # -----------------------------------------------------------------------
      tabItem(tabName = "howwedidit",
        fluidRow(
          column(12, tags$h2("3. How We Did It"))
        ),
        fluidRow(
          box(width = 6, title = "Class Imbalance — Before & After SMOTE",
            plotOutput("smote_plot", height = 300)
          ),
          box(width = 6, title = "Feature Selection — RF Importance",
            plotOutput("importance_plot", height = 300)
          )
        ),
        fluidRow(
          box(width = 12, title = "Key Technical Decisions",
            fluidRow(
              column(4,
                tags$h4(style="color:#F0C040", "Why SMOTE?"),
                tags$p("With only 3.2% bankrupt cases, any model trained on raw
                        data predicts 'not bankrupt' for everything. SMOTE generates
                        synthetic minority samples by interpolating between real
                        bankrupt companies and their nearest neighbours.")
              ),
              column(4,
                tags$h4(style="color:#F0C040", "Why Standardise?"),
                tags$p("Logistic Regression and Neural Networks are sensitive to
                        feature scale. A feature ranging 0–1 billion would dominate
                        one ranging 0–1. Standardisation (mean=0, SD=1) gives all
                        features equal footing. Scaler fit on training data only
                        to prevent data leakage.")
              ),
              column(4,
                tags$h4(style="color:#F0C040", "Why 10-fold CV?"),
                tags$p("Cross-validation trains and validates the model 10 times
                        on different data subsets, giving a more reliable performance
                        estimate than a single train/validate split and reducing
                        the risk of overfitting.")
              )
            )
          )
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 4: PROS & CONS
      # -----------------------------------------------------------------------
      tabItem(tabName = "proscons",
        fluidRow(
          column(12, tags$h2("4. Pros & Cons of Each Model"))
        ),
        fluidRow(
          column(4,
            tags$div(class="model-card card-lr",
              tags$h4(tags$span(class="badge-blue", "LR"),
                      " Logistic Regression"),
              tags$h4(style="color:#2ECC71; margin-top:15px", "✓ Advantages"),
              tags$ul(
                tags$li("Fully interpretable — coefficients show direction and
                         magnitude of each feature's effect"),
                tags$li("Fast to train — computationally inexpensive"),
                tags$li("Strong theoretical grounding in finance literature"),
                tags$li("Best recall (90.9%) — catches most bankruptcies"),
                tags$li("Highest AUC-ROC (0.941) — best overall discrimination")
              ),
              tags$h4(style="color:#E74C3C; margin-top:15px", "✗ Disadvantages"),
              tags$ul(
                tags$li("Assumes linear relationships — misses complex patterns"),
                tags$li("Sensitive to multicollinearity between features"),
                tags$li("158 false alarms — highest of all three models"),
                tags$li("Complete separation warnings indicate model limitations"),
                tags$li("Lowest precision (20.2%) — many false positives")
              )
            )
          ),
          column(4,
            tags$div(class="model-card card-rf",
              tags$h4(tags$span(class="badge-green", "RF"),
                      " Random Forest"),
              tags$h4(style="color:#2ECC71; margin-top:15px", "✓ Advantages"),
              tags$ul(
                tags$li("Best overall accuracy (93.7%) and F1 Score (41.9%)"),
                tags$li("Robust to outliers — important for financial data"),
                tags$li("Captures non-linear and interaction effects"),
                tags$li("Fewest false alarms (73) — most precise"),
                tags$li("Provides feature importance for interpretation")
              ),
              tags$h4(style="color:#E74C3C; margin-top:15px", "✗ Disadvantages"),
              tags$ul(
                tags$li("Less interpretable than Logistic Regression"),
                tags$li("Slower to train — requires hyperparameter tuning"),
                tags$li("Lower recall (70.5%) — misses more bankruptcies"),
                tags$li("Memory intensive with 500 trees"),
                tags$li("Gini importance biased against binary features
                         (Strobl et al., 2007)")
              )
            )
          ),
          column(4,
            tags$div(class="model-card card-nn",
              tags$h4(tags$span(class="badge-purple", "NN"),
                      " Neural Network"),
              tags$h4(style="color:#2ECC71; margin-top:15px", "✓ Advantages"),
              tags$ul(
                tags$li("Captures highly complex non-linear relationships"),
                tags$li("Architecture scales with data size"),
                tags$li("No assumptions about feature distributions"),
                tags$li("Competitive accuracy (92.7%) despite simpler framework"),
                tags$li("Industry standard for large-scale financial modelling")
              ),
              tags$h4(style="color:#E74C3C; margin-top:15px", "✗ Disadvantages"),
              tags$ul(
                tags$li("Complete black box — no interpretability"),
                tags$li("High training error (69.02) — incomplete convergence"),
                tags$li("Lowest AUC-ROC (0.853) — weakest discrimination"),
                tags$li("Requires large datasets to reach full potential"),
                tags$li("neuralnet uses basic backpropagation — no Adam optimiser")
              )
            )
          )
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 5: RESULTS
      # -----------------------------------------------------------------------
      tabItem(tabName = "results",
        fluidRow(
          column(12, tags$h2("5. Interpretation of Results"))
        ),
        fluidRow(
          box(width = 12, title = "Select Model to Inspect",
            selectInput("selected_model", label = NULL,
                        choices = c("Logistic Regression",
                                    "Random Forest",
                                    "Neural Network"),
                        selected = "Logistic Regression",
                        width = "300px")
          )
        ),
        fluidRow(
          box(width = 6, title = "Confusion Matrix",
            plotOutput("cm_plot", height = 320)
          ),
          box(width = 6, title = "ROC Curve",
            plotOutput("roc_plot", height = 320)
          )
        ),
        fluidRow(
          box(width = 12, title = "Performance Metrics",
            fluidRow(
              column(2, uiOutput("metric_accuracy")),
              column(2, uiOutput("metric_precision")),
              column(2, uiOutput("metric_recall")),
              column(2, uiOutput("metric_f1")),
              column(2, uiOutput("metric_specificity")),
              column(2, uiOutput("metric_auc"))
            )
          )
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 6: MODEL COMPARISON
      # -----------------------------------------------------------------------
      tabItem(tabName = "comparison",
        fluidRow(
          column(12, tags$h2("6. Model Comparison & Recommendation"))
        ),
        fluidRow(
          box(width = 7, title = "Combined ROC Curves",
            plotOutput("combined_roc", height = 380)
          ),
          box(width = 5, title = "Metrics Summary",
            DTOutput("metrics_table")
          )
        ),
        fluidRow(
          box(width = 12, title = "Bankruptcies Caught vs False Alarms",
            plotOutput("cm_summary_plot", height = 280)
          )
        ),
        fluidRow(
          box(width = 12, title = "Final Recommendation",
            fluidRow(
              column(6,
                tags$div(class="model-card card-rf",
                  tags$h4(tags$span(class="badge-green", "BEST OVERALL"),
                          " Random Forest"),
                  tags$p("Highest accuracy (93.7%), F1 Score (41.9%), and
                          specificity (94.5%). Fewest false alarms (73).
                          Best choice for general bankruptcy screening
                          where precision matters.")
                )
              ),
              column(6,
                tags$div(class="model-card card-lr",
                  tags$h4(tags$span(class="badge-blue", "BEST FOR RISK"),
                          " Logistic Regression"),
                  tags$p("Highest recall (90.9%) and AUC-ROC (0.941).
                          Catches 40 of 44 bankruptcies vs 31 for Random Forest.
                          Best choice for credit risk where missing a real
                          bankruptcy is more costly than a false alarm.")
                )
              )
            )
          )
        )
      )

    ) # end tabItems
  ) # end dashboardBody
) # end dashboardPage

# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {

  # --- Reactive: selected model results --------------------------------------
  selected_results <- reactive({
    switch(input$selected_model,
           "Logistic Regression" = lr_results,
           "Random Forest"       = rf_results,
           "Neural Network"      = nn_results)
  })

  selected_colour <- reactive({
    switch(input$selected_model,
           "Logistic Regression" = "#2980B9",
           "Random Forest"       = "#27AE60",
           "Neural Network"      = "#8E44AD")
  })

  # --- SMOTE plot ------------------------------------------------------------
  output$smote_plot <- renderPlot({
    before <- table(df_final$Bankrupt)
    after  <- table(train$Bankrupt)

    balance_df <- data.frame(
      Stage = rep(c("Before SMOTE", "After SMOTE"), each = 2),
      Class = rep(c("Not Bankrupt", "Bankrupt"), 2),
      Count = c(as.numeric(before), as.numeric(after))
    )
    balance_df$Stage <- factor(balance_df$Stage,
                               levels = c("Before SMOTE", "After SMOTE"))

    ggplot(balance_df, aes(x = Class, y = Count, fill = Class)) +
      geom_bar(stat = "identity", width = 0.5) +
      geom_text(aes(label = Count), vjust = -0.4, size = 4,
                fontface = "bold", colour = "white") +
      facet_wrap(~Stage) +
      scale_fill_manual(values = c("Not Bankrupt" = "#2ECC71",
                                   "Bankrupt"     = "#E74C3C")) +
      ylim(0, max(balance_df$Count) * 1.15) +
      labs(x = "", y = "Count") +
      theme_minimal(base_size = 11) +
      theme(legend.position  = "none",
            plot.background  = element_rect(fill = "#16213E", colour = NA),
            panel.background = element_rect(fill = "#16213E", colour = NA),
            panel.grid       = element_line(colour = "#2A2A4A"),
            text             = element_text(colour = "#E0E0E0"),
            strip.text       = element_text(colour = "#F0C040", size = 11))
  }, bg = "#16213E")

  # --- Feature importance plot -----------------------------------------------
  output$importance_plot <- renderPlot({
    top15 <- head(importance_df, 15)

    ggplot(top15, aes(x = reorder(Feature, MeanDecreaseGini),
                      y = MeanDecreaseGini)) +
      geom_bar(stat = "identity", fill = "#F0C040") +
      coord_flip() +
      labs(x = "", y = "Mean Decrease Gini") +
      theme_minimal(base_size = 9) +
      theme(plot.background  = element_rect(fill = "#16213E", colour = NA),
            panel.background = element_rect(fill = "#16213E", colour = NA),
            panel.grid       = element_line(colour = "#2A2A4A"),
            text             = element_text(colour = "#E0E0E0"))
  }, bg = "#16213E")

  # --- Confusion matrix plot -------------------------------------------------
  output$cm_plot <- renderPlot({
    cm_df <- as.data.frame(selected_results()$cm$table)
    colnames(cm_df) <- c("Predicted", "Actual", "Count")
    col <- selected_colour()

    ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
      geom_tile(colour = "white") +
      geom_text(aes(label = Count), size = 10,
                fontface = "bold", colour = "white") +
      scale_fill_gradient(low = col, high = "#0D0D1A") +
      labs(x = "Predicted Class", y = "Actual Class") +
      theme_minimal(base_size = 13) +
      theme(legend.position  = "none",
            plot.background  = element_rect(fill = "#16213E", colour = NA),
            panel.background = element_rect(fill = "#16213E", colour = NA),
            text             = element_text(colour = "#E0E0E0"),
            axis.text        = element_text(colour = "#E0E0E0", size = 12))
  }, bg = "#16213E")

  # --- ROC curve plot --------------------------------------------------------
  output$roc_plot <- renderPlot({
    roc_obj <- selected_results()$roc
    auc_val <- selected_results()$auc
    col     <- selected_colour()

    roc_df <- data.frame(
      FPR = 1 - roc_obj$specificities,
      TPR = roc_obj$sensitivities
    )

    ggplot(roc_df, aes(x = FPR, y = TPR)) +
      geom_line(colour = col, linewidth = 1.5) +
      geom_abline(linetype = "dashed", colour = "grey50") +
      annotate("text", x = 0.6, y = 0.15,
               label = paste0("AUC = ", auc_val),
               size = 5, colour = col, fontface = "bold") +
      labs(x = "False Positive Rate", y = "True Positive Rate") +
      theme_minimal(base_size = 12) +
      theme(plot.background  = element_rect(fill = "#16213E", colour = NA),
            panel.background = element_rect(fill = "#16213E", colour = NA),
            panel.grid       = element_line(colour = "#2A2A4A"),
            text             = element_text(colour = "#E0E0E0"))
  }, bg = "#16213E")

  # --- Metric boxes ----------------------------------------------------------
  metric_box <- function(value, label) {
    renderUI({
      tags$div(class = "metric-box",
               tags$div(class = "metric-value", value),
               tags$div(class = "metric-label", label))
    })
  }

  observeEvent(input$selected_model, {
    res <- selected_results()
    output$metric_accuracy    <- metric_box(
      paste0(res$metrics$Accuracy, "%"), "Accuracy")
    output$metric_precision   <- metric_box(
      paste0(res$metrics$Precision, "%"), "Precision")
    output$metric_recall      <- metric_box(
      paste0(res$metrics$Recall, "%"), "Recall")
    output$metric_f1          <- metric_box(
      paste0(res$metrics$F1_Score, "%"), "F1 Score")
    output$metric_specificity <- metric_box(
      paste0(res$metrics$Specificity, "%"), "Specificity")
    output$metric_auc         <- metric_box(
      res$metrics$AUC_ROC, "AUC-ROC")
  })

  # --- Combined ROC curve ----------------------------------------------------
  output$combined_roc <- renderPlot({
    roc_lr_df <- data.frame(
      FPR   = 1 - lr_results$roc$specificities,
      TPR   = lr_results$roc$sensitivities,
      Model = paste0("Logistic Regression (AUC=", lr_results$auc, ")")
    )
    roc_rf_df <- data.frame(
      FPR   = 1 - rf_results$roc$specificities,
      TPR   = rf_results$roc$sensitivities,
      Model = paste0("Random Forest (AUC=", rf_results$auc, ")")
    )
    roc_nn_df <- data.frame(
      FPR   = 1 - nn_results$roc$specificities,
      TPR   = nn_results$roc$sensitivities,
      Model = paste0("Neural Network (AUC=", nn_results$auc, ")")
    )

    roc_all <- bind_rows(roc_lr_df, roc_rf_df, roc_nn_df)

    model_colours <- setNames(
      c("#2980B9", "#27AE60", "#8E44AD"),
      c(paste0("Logistic Regression (AUC=", lr_results$auc, ")"),
        paste0("Random Forest (AUC=",       rf_results$auc, ")"),
        paste0("Neural Network (AUC=",      nn_results$auc, ")"))
    )

    ggplot(roc_all, aes(x = FPR, y = TPR, colour = Model)) +
      geom_line(linewidth = 1.4) +
      geom_abline(linetype = "dashed", colour = "grey50") +
      scale_colour_manual(values = model_colours) +
      labs(x = "False Positive Rate",
           y = "True Positive Rate", colour = "") +
      theme_minimal(base_size = 11) +
      theme(legend.position  = "bottom",
            legend.text      = element_text(size = 9, colour = "#E0E0E0"),
            plot.background  = element_rect(fill = "#16213E", colour = NA),
            panel.background = element_rect(fill = "#16213E", colour = NA),
            panel.grid       = element_line(colour = "#2A2A4A"),
            text             = element_text(colour = "#E0E0E0"))
  }, bg = "#16213E")

  # --- Metrics table ---------------------------------------------------------
  output$metrics_table <- renderDT({
    datatable(
      comparison_df,
      options  = list(dom = "t", pageLength = 3,
                      initComplete = JS(
                        "function(settings, json) {",
                        "$(this.api().table().header()).css({",
                        "'background-color': '#0F3460',",
                        "'color': '#F0C040'});",
                        "}"
                      )),
      rownames = FALSE,
      class    = "compact"
    ) %>%
      formatStyle(columns = 1:7,
                  backgroundColor = "#16213E",
                  color           = "#E0E0E0")
  })

  # --- Confusion matrix summary bar chart ------------------------------------
  output$cm_summary_plot <- renderPlot({
    plot_df <- data.frame(
      Model  = rep(cm_summary$Model, 2),
      Type   = c(rep("Bankruptcies Caught (TP)", 3),
                 rep("False Alarms (FP)", 3)),
      Count  = c(cm_summary$True_Positive,
                 cm_summary$False_Positive)
    )

    ggplot(plot_df, aes(x = Model, y = Count, fill = Type)) +
      geom_bar(stat = "identity", position = "dodge", width = 0.6) +
      geom_text(aes(label = Count),
                position = position_dodge(width = 0.6),
                vjust = -0.4, size = 4.5,
                fontface = "bold", colour = "white") +
      scale_fill_manual(values = c("Bankruptcies Caught (TP)" = "#2ECC71",
                                   "False Alarms (FP)"        = "#E74C3C")) +
      ylim(0, max(plot_df$Count) * 1.15) +
      labs(x = "", y = "Count", fill = "") +
      theme_minimal(base_size = 12) +
      theme(legend.position  = "bottom",
            legend.text      = element_text(colour = "#E0E0E0"),
            plot.background  = element_rect(fill = "#16213E", colour = NA),
            panel.background = element_rect(fill = "#16213E", colour = NA),
            panel.grid       = element_line(colour = "#2A2A4A"),
            text             = element_text(colour = "#E0E0E0"),
            axis.text.x      = element_text(colour = "#F0C040",
                                            size = 11, face = "bold"))
  }, bg = "#16213E")

}

# =============================================================================
# RUN APP
# =============================================================================

shinyApp(ui = ui, server = server)