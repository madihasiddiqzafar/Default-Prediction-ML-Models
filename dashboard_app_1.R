# =============================================================================
# ASSIGNMENT 2 — SHINY DASHBOARD
# Big Data Module - Bankruptcy Prediction
# University of Birmingham Business School
# =============================================================================

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

comparison_df <- bind_rows(
  lr_results$metrics,
  rf_results$metrics,
  nn_results$metrics
)

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

importance_df <- data.frame(
  Feature          = rownames(importance(rf_results$model)),
  MeanDecreaseGini = importance(rf_results$model)[, "MeanDecreaseGini"],
  row.names        = NULL
) %>% arrange(desc(MeanDecreaseGini))

# =============================================================================
# CONSTANTS
# =============================================================================

FEATURE_NAMES <- c(
  "Net.Income.to.Total.Assets",
  "Retained.Earnings.to.Total.Assets",
  "Persistent.EPS.in.the.Last.Four.Seasons",
  "ROA.B..before.interest.and.depreciation.after.tax",
  "ROA.A..before.interest.and...after.tax",
  "ROA.C..before.interest.and.depreciation.before.interest",
  "Continuous.interest.rate..after.tax.",
  "Per.Share.Net.profit.before.tax..Yuan.Â..",
  "Total.debt.Total.net.worth",
  "Net.profit.before.tax.Paid.in.capital",
  "Borrowing.dependency",
  "Total.income.Total.expense",
  "Debt.ratio..",
  "Equity.to.Liability",
  "Net.worth.Assets",
  "Interest.Expense.Ratio",
  "Cash.Turnover.Rate",
  "Pre.tax.net.Interest.Rate",
  "Interest.bearing.debt.interest.rate",
  "Quick.Ratio",
  "Liability.Assets.Flag",
  "Net.Income.Flag"
)

BINARY_COLS   <- c("Liability.Assets.Flag", "Net.Income.Flag")
NUMERIC_FEATS <- setdiff(FEATURE_NAMES, BINARY_COLS)

# Load scaler once at startup
SCALER <- readRDS("scaler.rds")

# =============================================================================
# PRESET VALUES
# Real rows from train_scaled.rds — guaranteed to work with all three models.
# Row 1  = bankrupt  (NN output 0.99)
# Row 5282 = healthy (NN output ~0.00)
#
# NN was trained with factor levels: 1 = bankrupt, 2 = not bankrupt
# NN single output node: high value (~1) = bankrupt, low value (~0) = not bankrupt
# So nn_prob = raw NN output directly (no flip needed)
# =============================================================================

# Scaled values (fed to NN and LR)
BANKRUPT_SCALED <- c(-0.240635, -0.029318, -0.349350, -0.554264, -0.442196,
                     -0.629484,  0.018473, -0.317369, -0.040537, -0.339350,
                     -0.078941, -0.034782, -0.071728, -0.229430,  0.071728,
                     -0.081081, -0.876168,  0.019532, -0.118730, -0.053308,
                      0.000000,  1.000000)

HEALTHY_SCALED  <- c( 0.644439,  0.472348,  0.431560,  0.602143,  0.662817,
                      0.578843,  0.065410,  0.392601, -0.040537,  0.419222,
                     -0.207245,  0.002169, -0.661889, -0.020114,  0.661889,
                      0.049610, -0.606889,  0.068102, -0.118730, -0.053308,
                      0.000000,  1.000000)

# Back-calculate raw display values: raw = scaled * std + mean

BANKRUPT_RAW <- BANKRUPT_SCALED
HEALTHY_RAW  <- HEALTHY_SCALED

# =============================================================================
# UI
# =============================================================================

ui <- dashboardPage(
  skin = "black",

  dashboardHeader(
    title = span(style = "font-family:'Georgia',serif;font-size:16px;color:#F0C040;",
                 "Bankruptcy Prediction — Big Data"),
    titleWidth = 320
  ),

  dashboardSidebar(
    width = 260,
    tags$div(style = "padding:15px 10px 5px 15px;",
      tags$p(style = "color:#AAA;font-size:11px;margin:0;", "University of Birmingham"),
      tags$p(style = "color:#AAA;font-size:11px;margin:0;", "Big Data — Assignment 2"),
      tags$hr(style = "border-color:#444;margin:10px 0;")
    ),
    sidebarMenu(
      menuItem("1. Background",       tabName = "background",  icon = icon("book-open")),
      menuItem("2. Steps Performed",  tabName = "steps",       icon = icon("list-ol")),
      menuItem("3. How We Did It",    tabName = "howwedidit",  icon = icon("code")),
      menuItem("4. Pros & Cons",      tabName = "proscons",    icon = icon("balance-scale")),
      menuItem("5. Results",          tabName = "results",     icon = icon("chart-bar")),
      menuItem("6. Model Comparison", tabName = "comparison",  icon = icon("trophy")),
      menuItem("7. Live Prediction",  tabName = "prediction",  icon = icon("search-dollar"))
    )
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      body,.content-wrapper,.main-sidebar,.sidebar{font-family:'Georgia',serif;background-color:#1A1A2E;}
      .content-wrapper{background-color:#1A1A2E;}
      .box{background-color:#16213E;border-top:3px solid #F0C040;border-radius:6px;color:#E0E0E0;}
      .box-header{color:#F0C040!important;} .box-title{color:#F0C040!important;font-size:15px;}
      h2{color:#F0C040;font-family:'Georgia',serif;} h3{color:#E0E0E0;font-family:'Georgia',serif;}
      h4{color:#F0C040;} p,li{color:#C8C8C8;line-height:1.7;}
      .metric-box{background:#0F3460;border-left:4px solid #F0C040;border-radius:5px;padding:15px 20px;margin:8px 0;}
      .metric-value{font-size:28px;font-weight:bold;color:#F0C040;}
      .metric-label{font-size:12px;color:#AAA;text-transform:uppercase;letter-spacing:1px;}
      .model-card{background:#0F3460;border-radius:8px;padding:20px;margin:10px 0;border-left:5px solid;}
      .card-lr{border-color:#2980B9;} .card-rf{border-color:#27AE60;} .card-nn{border-color:#8E44AD;}
      .step-card{background:#0F3460;border-radius:6px;padding:15px 20px;margin:8px 0;border-left:4px solid #F0C040;}
      .badge-blue{background:#2980B9;color:#fff;padding:3px 10px;border-radius:12px;font-size:11px;}
      .badge-green{background:#27AE60;color:#fff;padding:3px 10px;border-radius:12px;font-size:11px;}
      .badge-purple{background:#8E44AD;color:#fff;padding:3px 10px;border-radius:12px;font-size:11px;}
      .badge-gold{background:#F0C040;color:#000;padding:3px 10px;border-radius:12px;font-size:11px;}
      table.dataTable{color:#E0E0E0!important;} .dataTables_wrapper{color:#E0E0E0;}
      .main-header .logo{background-color:#0D0D1A!important;}
      .main-header .navbar{background-color:#0D0D1A!important;}
      .main-sidebar{background-color:#0D0D1A!important;}
      .sidebar-menu>li>a{color:#CCC!important;}
      .sidebar-menu>li.active>a{color:#F0C040!important;border-left:3px solid #F0C040;}
    "))),

    tabItems(

      # -----------------------------------------------------------------------
      # TAB 1: BACKGROUND
      # -----------------------------------------------------------------------
      tabItem(tabName = "background",
        fluidRow(column(12,
          tags$h2("1. Background of Methods Used"),
          tags$p("This project applies machine learning and deep learning techniques to predict
                  corporate bankruptcy using financial ratio data from 6,819 Taiwanese companies.")
        )),
        fluidRow(
          column(4, tags$div(class="model-card card-lr",
            tags$h4(tags$span(class="badge-blue","Model 1")," Logistic Regression"),
            tags$p("A statistical method that models the probability of a binary outcome as a
                    function of predictor variables. Standard baseline in credit risk modelling
                    (Altman & Sabato, 2007). Produces fully interpretable coefficients."),
            tags$p(tags$strong(style="color:#2980B9","Type:"),
                   tags$span(style="color:#CCC"," Supervised | Linear | Parametric"))
          )),
          column(4, tags$div(class="model-card card-rf",
            tags$h4(tags$span(class="badge-green","Model 2")," Random Forest"),
            tags$p("An ensemble method training hundreds of decision trees on random subsets,
                    then combining predictions. Robust to outliers and captures non-linear
                    relationships (Breiman, 2001)."),
            tags$p(tags$strong(style="color:#27AE60","Type:"),
                   tags$span(style="color:#CCC"," Supervised | Non-linear | Ensemble"))
          )),
          column(4, tags$div(class="model-card card-nn",
            tags$h4(tags$span(class="badge-purple","Model 3")," Neural Network"),
            tags$p("A deep learning model with two hidden layers (16 and 8 nodes) learning
                    complex non-linear representations through backpropagation (Tam & Kiang, 1992)."),
            tags$p(tags$strong(style="color:#8E44AD","Type:"),
                   tags$span(style="color:#CCC"," Supervised | Deep Learning | 2 Hidden Layers"))
          ))
        ),
        fluidRow(box(width=12,title="Dataset Overview",
          fluidRow(
            column(3,tags$div(class="metric-box",tags$div(class="metric-value","6,819"),tags$div(class="metric-label","Companies"))),
            column(3,tags$div(class="metric-box",tags$div(class="metric-value","95"),   tags$div(class="metric-label","Original Features"))),
            column(3,tags$div(class="metric-box",tags$div(class="metric-value","22"),   tags$div(class="metric-label","Selected Features"))),
            column(3,tags$div(class="metric-box",tags$div(class="metric-value","3.2%"),tags$div(class="metric-label","Bankruptcy Rate")))
          )
        ))
      ),

      # -----------------------------------------------------------------------
      # TAB 2: STEPS PERFORMED
      # -----------------------------------------------------------------------
      tabItem(tabName="steps",
        fluidRow(column(12,tags$h2("2. Steps Performed"))),
        fluidRow(
          column(6,
            tags$div(class="step-card",tags$h4("Step 1 — Data Quality Control"),tags$ul(
              tags$li("Assessed dataset size: 6,819 rows x 96 columns"),
              tags$li("Confirmed zero missing values across all features"),
              tags$li("Identified severe class imbalance (96.8% vs 3.2%)"),
              tags$li("Detected outliers using the IQR method"),
              tags$li("Visualised feature distributions and correlations")
            )),
            tags$div(class="step-card",tags$h4("Step 2 — Feature Selection"),tags$ul(
              tags$li("Removed near-zero variance features"),
              tags$li("Ranked remaining features by Random Forest Gini importance"),
              tags$li("Selected top 20 features from importance ranking"),
              tags$li("Force-included 2 binary distress flags (Altman, 1968; Zmijewski, 1984)"),
              tags$li("Final feature set: 22 variables")
            )),
            tags$div(class="step-card",tags$h4("Step 3 — Data Preparation"),tags$ul(
              tags$li("Stratified 80/20 train/test split"),
              tags$li("Applied SMOTE to training set only (10,560 balanced rows)"),
              tags$li("Standardised continuous features (mean=0, SD=1)"),
              tags$li("Binary flags left unscaled to preserve meaning")
            ))
          ),
          column(6,
            tags$div(class="step-card",tags$h4("Step 4 — Model Development"),tags$ul(
              tags$li("Trained Logistic Regression with 10-fold cross-validation"),
              tags$li("Tuned Random Forest mtry via 5-fold CV (best mtry = 4)"),
              tags$li("Trained Neural Network: 2 hidden layers (16 -> 8 nodes)"),
              tags$li("All models evaluated on the same held-out test set")
            )),
            tags$div(class="step-card",tags$h4("Step 5 — Evaluation & Comparison"),tags$ul(
              tags$li("Computed confusion matrix for each model"),
              tags$li("Calculated Accuracy, Precision, Recall, F1, Specificity"),
              tags$li("Generated AUC-ROC scores and curves"),
              tags$li("Produced combined ROC plot for direct comparison"),
              tags$li("Identified best model per metric and overall")
            )),
            tags$div(class="step-card",tags$h4("Step 6 — Interpretation"),tags$ul(
              tags$li("Interpreted feature importance in financial context"),
              tags$li("Evaluated models against real-world decision use cases"),
              tags$li("Discussed ethical considerations and GDPR implications"),
              tags$li("Acknowledged limitations of methodology")
            ))
          )
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 3: HOW WE DID IT
      # -----------------------------------------------------------------------
      tabItem(tabName="howwedidit",
        fluidRow(column(12,tags$h2("3. How We Did It"))),
        fluidRow(
          box(width=6,title="Class Imbalance — Before & After SMOTE",plotOutput("smote_plot",height=300)),
          box(width=6,title="Feature Selection — RF Importance",plotOutput("importance_plot",height=300))
        ),
        fluidRow(box(width=12,title="Key Technical Decisions",
          fluidRow(
            column(4,tags$h4(style="color:#F0C040","Why SMOTE?"),
              tags$p("With only 3.2% bankrupt cases, any model trained on raw data predicts
                      'not bankrupt' for everything. SMOTE generates synthetic minority samples
                      by interpolating between real bankrupt companies and their nearest neighbours.")),
            column(4,tags$h4(style="color:#F0C040","Why Standardise?"),
              tags$p("Logistic Regression and Neural Networks are sensitive to feature scale.
                      Standardisation (mean=0, SD=1) gives all features equal footing.
                      Scaler fit on training data only to prevent data leakage.")),
            column(4,tags$h4(style="color:#F0C040","Why 10-fold CV?"),
              tags$p("Cross-validation trains and validates the model 10 times on different
                      data subsets, giving a more reliable performance estimate and reducing
                      the risk of overfitting."))
          )
        ))
      ),

      # -----------------------------------------------------------------------
      # TAB 4: PROS & CONS
      # -----------------------------------------------------------------------
      tabItem(tabName="proscons",
        fluidRow(column(12,tags$h2("4. Pros & Cons of Each Model"))),
        fluidRow(
          column(4,tags$div(class="model-card card-lr",
            tags$h4(tags$span(class="badge-blue","LR")," Logistic Regression"),
            tags$h4(style="color:#2ECC71;margin-top:15px","Advantages"),
            tags$ul(
              tags$li("Fully interpretable — coefficients show direction and magnitude"),
              tags$li("Fast to train — computationally inexpensive"),
              tags$li("Strong theoretical grounding in finance literature"),
              tags$li("Best recall (90.9%) — catches most bankruptcies"),
              tags$li("Highest AUC-ROC (0.941) — best overall discrimination")
            ),
            tags$h4(style="color:#E74C3C;margin-top:15px","Disadvantages"),
            tags$ul(
              tags$li("Assumes linear relationships — misses complex patterns"),
              tags$li("Sensitive to multicollinearity between features"),
              tags$li("158 false alarms — highest of all three models"),
              tags$li("Complete separation warnings indicate model limitations"),
              tags$li("Lowest precision (20.2%) — many false positives")
            )
          )),
          column(4,tags$div(class="model-card card-rf",
            tags$h4(tags$span(class="badge-green","RF")," Random Forest"),
            tags$h4(style="color:#2ECC71;margin-top:15px","Advantages"),
            tags$ul(
              tags$li("Best overall accuracy (93.7%) and F1 Score (41.9%)"),
              tags$li("Robust to outliers — important for financial data"),
              tags$li("Captures non-linear and interaction effects"),
              tags$li("Fewest false alarms (73) — most precise"),
              tags$li("Provides feature importance for interpretation")
            ),
            tags$h4(style="color:#E74C3C;margin-top:15px","Disadvantages"),
            tags$ul(
              tags$li("Less interpretable than Logistic Regression"),
              tags$li("Slower to train — requires hyperparameter tuning"),
              tags$li("Lower recall (70.5%) — misses more bankruptcies"),
              tags$li("Memory intensive with 500 trees"),
              tags$li("Gini importance biased against binary features (Strobl et al., 2007)")
            )
          )),
          column(4,tags$div(class="model-card card-nn",
            tags$h4(tags$span(class="badge-purple","NN")," Neural Network"),
            tags$h4(style="color:#2ECC71;margin-top:15px","Advantages"),
            tags$ul(
              tags$li("Captures highly complex non-linear relationships"),
              tags$li("Architecture scales with data size"),
              tags$li("No assumptions about feature distributions"),
              tags$li("Competitive accuracy (92.7%) despite simpler framework"),
              tags$li("Industry standard for large-scale financial modelling")
            ),
            tags$h4(style="color:#E74C3C;margin-top:15px","Disadvantages"),
            tags$ul(
              tags$li("Complete black box — no interpretability"),
              tags$li("High training error (69.02) — incomplete convergence"),
              tags$li("Lowest AUC-ROC (0.853) — weakest discrimination"),
              tags$li("Requires large datasets to reach full potential"),
              tags$li("neuralnet uses basic backpropagation — no Adam optimiser")
            )
          ))
        )
      ),

      # -----------------------------------------------------------------------
      # TAB 5: RESULTS
      # -----------------------------------------------------------------------
      tabItem(tabName="results",
        fluidRow(column(12,tags$h2("5. Interpretation of Results"))),
        fluidRow(box(width=12,title="Select Model to Inspect",
          selectInput("selected_model",label=NULL,
                      choices=c("Logistic Regression","Random Forest","Neural Network"),
                      selected="Logistic Regression",width="300px")
        )),
        fluidRow(
          box(width=6,title="Confusion Matrix",plotOutput("cm_plot",height=320)),
          box(width=6,title="ROC Curve",       plotOutput("roc_plot",height=320))
        ),
        fluidRow(box(width=12,title="Performance Metrics",
          fluidRow(
            column(2,uiOutput("metric_accuracy")),
            column(2,uiOutput("metric_precision")),
            column(2,uiOutput("metric_recall")),
            column(2,uiOutput("metric_f1")),
            column(2,uiOutput("metric_specificity")),
            column(2,uiOutput("metric_auc"))
          )
        ))
      ),

      # -----------------------------------------------------------------------
      # TAB 6: MODEL COMPARISON
      # -----------------------------------------------------------------------
      tabItem(tabName="comparison",
        fluidRow(column(12,tags$h2("6. Model Comparison & Recommendation"))),
        fluidRow(
          box(width=7,title="Combined ROC Curves",plotOutput("combined_roc",height=380)),
          box(width=5,title="Metrics Summary",    DTOutput("metrics_table"))
        ),
        fluidRow(box(width=12,title="Bankruptcies Caught vs False Alarms",
          plotOutput("cm_summary_plot",height=280)
        )),
        fluidRow(box(width=12,title="Final Recommendation",
          fluidRow(
            column(6,tags$div(class="model-card card-rf",
              tags$h4(tags$span(class="badge-green","BEST OVERALL")," Random Forest"),
              tags$p("Highest accuracy (93.7%), F1 Score (41.9%), and specificity (94.5%).
                      Fewest false alarms (73). Best for general bankruptcy screening.")
            )),
            column(6,tags$div(class="model-card card-lr",
              tags$h4(tags$span(class="badge-blue","BEST FOR RISK")," Logistic Regression"),
              tags$p("Highest recall (90.9%) and AUC-ROC (0.941). Catches 40 of 44 bankruptcies
                      vs 31 for RF. Best for credit risk where missing a bankruptcy is costly.")
            ))
          )
        ))
      ),

      # -----------------------------------------------------------------------
      # TAB 7: LIVE PREDICTION
      # -----------------------------------------------------------------------
      tabItem(tabName="prediction",
        fluidRow(column(12,
          tags$h2("7. Live Bankruptcy Prediction Tool"),
          tags$p("Select a stakeholder perspective and load a preset company or enter custom values.
                  All three models predict simultaneously.")
        )),
        fluidRow(box(width=12,title="Stakeholder & Company Selection",
          fluidRow(
            column(4,selectInput("stakeholder","Stakeholder Perspective:",
              choices=c("Lender / Credit Analyst"="lender",
                        "Investor / Portfolio Mgr"="investor",
                        "Regulator / Government"="regulator"),width="100%")),
            column(4,selectInput("preset_company","Load Preset Company:",
              choices=c("Custom (enter manually)"="custom",
                        "Sample Bankrupt Company"="bankrupt",
                        "Sample Healthy Company"="healthy"),width="100%")),
            column(4,tags$br(),
              actionButton("predict_btn","Run Prediction",
                style="background-color:#F0C040;color:#000;font-weight:bold;width:100%;
                       border:none;padding:10px;border-radius:5px;font-size:14px;margin-top:5px;"))
          )
        )),
        fluidRow(
          box(width=6,title="Key Financial Ratios (standardised scale)",
            tags$p(style="color:#AAA;font-size:11px;margin-bottom:10px;",
                   "Values are original 0-1 normalised ratios. Presets use real training data rows."),
            fluidRow(
              column(6,
                numericInput("inp_f1",  "Net Income to Total Assets",                       value=0.763,min=0,max=1,step=0.001),
                numericInput("inp_f2",  "Retained Earnings to Total Assets",                value=0.920,min=0,max=1,step=0.001),
                numericInput("inp_f3",  "Persistent EPS (Last Four Seasons)",               value=0.199,min=0,max=1,step=0.001),
                numericInput("inp_f4",  "ROA B (before interest & depreciation after tax)", value=0.469,min=0,max=1,step=0.001),
                numericInput("inp_f5",  "ROA A (before interest after tax)",                value=0.475,min=0,max=1,step=0.001),
                numericInput("inp_f6",  "ROA C (before interest & depreciation)",           value=0.444,min=0,max=1,step=0.001),
                numericInput("inp_f7",  "Continuous Interest Rate (after tax)",             value=0.212,min=0,max=1,step=0.001),
                numericInput("inp_f8",  "Per Share Net Profit before Tax",                  value=0.460,min=0,max=1,step=0.001),
                numericInput("inp_f9",  "Total Debt / Total Net Worth",                     value=0.366,min=0,max=1,step=0.001),
                numericInput("inp_f10", "Net Profit before Tax / Paid-in Capital",          value=0.453,min=0,max=1,step=0.001)
              ),
              column(6,
                numericInput("inp_f11", "Borrowing Dependency",                             value=0.358,min=0,max=1,step=0.001),
                numericInput("inp_f12", "Total Income / Total Expense",                     value=0.503,min=0,max=1,step=0.001),
                numericInput("inp_f13", "Debt Ratio",                                       value=0.361,min=0,max=1,step=0.001),
                numericInput("inp_f14", "Equity to Liability",                              value=0.493,min=0,max=1,step=0.001),
                numericInput("inp_f15", "Net Worth / Assets",                               value=0.639,min=0,max=1,step=0.001),
                numericInput("inp_f16", "Interest Expense Ratio",                           value=0.495,min=0,max=1,step=0.001),
                numericInput("inp_f17", "Cash Turnover Rate",                               value=0.482,min=0,max=1,step=0.001),
                numericInput("inp_f18", "Pre-tax Net Interest Rate",                        value=0.501,min=0,max=1,step=0.001),
                numericInput("inp_f19", "Interest-bearing Debt Interest Rate",              value=0.372,min=0,max=1,step=0.001),
                numericInput("inp_f20", "Quick Ratio",                                      value=0.494,min=0,max=1,step=0.001),
                tags$br(),
                tags$h4(style="color:#F0C040","Binary Distress Flags"),
                selectInput("inp_liab_flag",  "Liability-Assets Flag (Liabilities > Assets?)",
                            choices=c("No (0)"=0,"Yes (1)"=1),selected=0,width="100%"),
                selectInput("inp_income_flag","Net Income Flag (Negative 2+ years?)",
                            choices=c("No (0)"=0,"Yes (1)"=1),selected=1,width="100%")
              )
            )
          ),
          box(width=6,title="Prediction Results",
            uiOutput("stakeholder_context"),
            tags$hr(style="border-color:#2A2A4A"),
            tags$h4(style="color:#F0C040;margin-top:15px","Bankruptcy Probability — All Three Models"),
            uiOutput("pred_lr"),
            uiOutput("pred_rf"),
            uiOutput("pred_nn"),
            tags$hr(style="border-color:#2A2A4A"),
            uiOutput("pred_verdict"),
            tags$hr(style="border-color:#2A2A4A"),
            uiOutput("stakeholder_advice")
          )
        )
      )

    )
  )
)

# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {

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

  # SMOTE plot
  output$smote_plot <- renderPlot({
    before <- table(df_final$Bankrupt)
    after  <- table(train$Bankrupt)
    balance_df <- data.frame(
      Stage = rep(c("Before SMOTE","After SMOTE"),each=2),
      Class = rep(c("Not Bankrupt","Bankrupt"),2),
      Count = c(as.numeric(before),as.numeric(after))
    )
    balance_df$Stage <- factor(balance_df$Stage,levels=c("Before SMOTE","After SMOTE"))
    ggplot(balance_df,aes(x=Class,y=Count,fill=Class))+
      geom_bar(stat="identity",width=0.5)+
      geom_text(aes(label=Count),vjust=-0.4,size=4,fontface="bold",colour="white")+
      facet_wrap(~Stage)+
      scale_fill_manual(values=c("Not Bankrupt"="#2ECC71","Bankrupt"="#E74C3C"))+
      ylim(0,max(balance_df$Count)*1.15)+labs(x="",y="Count")+
      theme_minimal(base_size=11)+
      theme(legend.position="none",
            plot.background=element_rect(fill="#16213E",colour=NA),
            panel.background=element_rect(fill="#16213E",colour=NA),
            panel.grid=element_line(colour="#2A2A4A"),
            text=element_text(colour="#E0E0E0"),
            strip.text=element_text(colour="#F0C040",size=11))
  },bg="#16213E")

  # Feature importance plot
  output$importance_plot <- renderPlot({
    top15 <- head(importance_df,15)
    ggplot(top15,aes(x=reorder(Feature,MeanDecreaseGini),y=MeanDecreaseGini))+
      geom_bar(stat="identity",fill="#F0C040")+coord_flip()+
      labs(x="",y="Mean Decrease Gini")+theme_minimal(base_size=9)+
      theme(plot.background=element_rect(fill="#16213E",colour=NA),
            panel.background=element_rect(fill="#16213E",colour=NA),
            panel.grid=element_line(colour="#2A2A4A"),
            text=element_text(colour="#E0E0E0"))
  },bg="#16213E")

  # Confusion matrix plot
  output$cm_plot <- renderPlot({
    cm_df <- as.data.frame(selected_results()$cm$table)
    colnames(cm_df) <- c("Predicted","Actual","Count")
    col <- selected_colour()
    ggplot(cm_df,aes(x=Predicted,y=Actual,fill=Count))+
      geom_tile(colour="white")+
      geom_text(aes(label=Count),size=10,fontface="bold",colour="white")+
      scale_fill_gradient(low=col,high="#0D0D1A")+
      labs(x="Predicted Class",y="Actual Class")+theme_minimal(base_size=13)+
      theme(legend.position="none",
            plot.background=element_rect(fill="#16213E",colour=NA),
            panel.background=element_rect(fill="#16213E",colour=NA),
            text=element_text(colour="#E0E0E0"),
            axis.text=element_text(colour="#E0E0E0",size=12))
  },bg="#16213E")

  # ROC curve
  output$roc_plot <- renderPlot({
    roc_obj <- selected_results()$roc
    auc_val <- selected_results()$auc
    col     <- selected_colour()
    roc_df  <- data.frame(FPR=1-roc_obj$specificities,TPR=roc_obj$sensitivities)
    ggplot(roc_df,aes(x=FPR,y=TPR))+
      geom_line(colour=col,linewidth=1.5)+
      geom_abline(linetype="dashed",colour="grey50")+
      annotate("text",x=0.6,y=0.15,label=paste0("AUC = ",auc_val),
               size=5,colour=col,fontface="bold")+
      labs(x="False Positive Rate",y="True Positive Rate")+theme_minimal(base_size=12)+
      theme(plot.background=element_rect(fill="#16213E",colour=NA),
            panel.background=element_rect(fill="#16213E",colour=NA),
            panel.grid=element_line(colour="#2A2A4A"),
            text=element_text(colour="#E0E0E0"))
  },bg="#16213E")

  # Metric boxes
  metric_box_ui <- function(value,label){
    tags$div(class="metric-box",
             tags$div(class="metric-value",value),
             tags$div(class="metric-label",label))
  }
  output$metric_accuracy    <- renderUI(metric_box_ui(paste0(selected_results()$metrics$Accuracy,   "%"),"Accuracy"))
  output$metric_precision   <- renderUI(metric_box_ui(paste0(selected_results()$metrics$Precision,  "%"),"Precision"))
  output$metric_recall      <- renderUI(metric_box_ui(paste0(selected_results()$metrics$Recall,     "%"),"Recall"))
  output$metric_f1          <- renderUI(metric_box_ui(paste0(selected_results()$metrics$F1_Score,   "%"),"F1 Score"))
  output$metric_specificity <- renderUI(metric_box_ui(paste0(selected_results()$metrics$Specificity,"%"),"Specificity"))
  output$metric_auc         <- renderUI(metric_box_ui(selected_results()$metrics$AUC_ROC,           "AUC-ROC"))

  # Combined ROC
  output$combined_roc <- renderPlot({
    make_roc <- function(res,lbl) data.frame(
      FPR=1-res$roc$specificities, TPR=res$roc$sensitivities,
      Model=paste0(lbl," (AUC=",res$auc,")")
    )
    roc_all <- bind_rows(make_roc(lr_results,"Logistic Regression"),
                         make_roc(rf_results,"Random Forest"),
                         make_roc(nn_results,"Neural Network"))
    cols <- setNames(c("#2980B9","#27AE60","#8E44AD"),unique(roc_all$Model))
    ggplot(roc_all,aes(x=FPR,y=TPR,colour=Model))+
      geom_line(linewidth=1.4)+geom_abline(linetype="dashed",colour="grey50")+
      scale_colour_manual(values=cols)+
      labs(x="False Positive Rate",y="True Positive Rate",colour="")+
      theme_minimal(base_size=11)+
      theme(legend.position="bottom",legend.text=element_text(size=9,colour="#E0E0E0"),
            plot.background=element_rect(fill="#16213E",colour=NA),
            panel.background=element_rect(fill="#16213E",colour=NA),
            panel.grid=element_line(colour="#2A2A4A"),
            text=element_text(colour="#E0E0E0"))
  },bg="#16213E")

  # Metrics table
  output$metrics_table <- renderDT({
    datatable(comparison_df,
      options=list(dom="t",pageLength=3,
        initComplete=JS("function(settings,json){",
          "$(this.api().table().header()).css({",
          "'background-color':'#0F3460','color':'#F0C040'});","}")),
      rownames=FALSE,class="compact"
    ) %>% formatStyle(columns=1:7,backgroundColor="#16213E",color="#E0E0E0")
  })

  # CM summary bar chart
  output$cm_summary_plot <- renderPlot({
    plot_df <- data.frame(
      Model=rep(cm_summary$Model,2),
      Type=c(rep("Bankruptcies Caught (TP)",3),rep("False Alarms (FP)",3)),
      Count=c(cm_summary$True_Positive,cm_summary$False_Positive)
    )
    ggplot(plot_df,aes(x=Model,y=Count,fill=Type))+
      geom_bar(stat="identity",position="dodge",width=0.6)+
      geom_text(aes(label=Count),position=position_dodge(width=0.6),
                vjust=-0.4,size=4.5,fontface="bold",colour="white")+
      scale_fill_manual(values=c("Bankruptcies Caught (TP)"="#2ECC71","False Alarms (FP)"="#E74C3C"))+
      ylim(0,max(plot_df$Count)*1.15)+labs(x="",y="Count",fill="")+
      theme_minimal(base_size=12)+
      theme(legend.position="bottom",legend.text=element_text(colour="#E0E0E0"),
            plot.background=element_rect(fill="#16213E",colour=NA),
            panel.background=element_rect(fill="#16213E",colour=NA),
            panel.grid=element_line(colour="#2A2A4A"),
            text=element_text(colour="#E0E0E0"),
            axis.text.x=element_text(colour="#F0C040",size=11,face="bold"))
  },bg="#16213E")

  # =========================================================================
  # TAB 7: PREDICTION
  # =========================================================================

  # Load presets
  observeEvent(input$preset_company, {
    v <- if (input$preset_company == "bankrupt") BANKRUPT_RAW
         else if (input$preset_company == "healthy") HEALTHY_RAW
         else return()
    updateNumericInput(session,"inp_f1", value=v[1]);  updateNumericInput(session,"inp_f2", value=v[2])
    updateNumericInput(session,"inp_f3", value=v[3]);  updateNumericInput(session,"inp_f4", value=v[4])
    updateNumericInput(session,"inp_f5", value=v[5]);  updateNumericInput(session,"inp_f6", value=v[6])
    updateNumericInput(session,"inp_f7", value=v[7]);  updateNumericInput(session,"inp_f8", value=v[8])
    updateNumericInput(session,"inp_f9", value=v[9]);  updateNumericInput(session,"inp_f10",value=v[10])
    updateNumericInput(session,"inp_f11",value=v[11]); updateNumericInput(session,"inp_f12",value=v[12])
    updateNumericInput(session,"inp_f13",value=v[13]); updateNumericInput(session,"inp_f14",value=v[14])
    updateNumericInput(session,"inp_f15",value=v[15]); updateNumericInput(session,"inp_f16",value=v[16])
    updateNumericInput(session,"inp_f17",value=v[17]); updateNumericInput(session,"inp_f18",value=v[18])
    updateNumericInput(session,"inp_f19",value=v[19]); updateNumericInput(session,"inp_f20",value=v[20])
    updateSelectInput(session,"inp_liab_flag",  selected=v[21])
    updateSelectInput(session,"inp_income_flag",selected=v[22])
  })

  # Stakeholder context
  output$stakeholder_context <- renderUI({
    ctx <- switch(input$stakeholder,
      "lender"   = list(title="Lender / Credit Analyst View",colour="#2980B9",
        text="As a lender, RECALL is your priority. Missing a real bankruptcy means extending
              credit to a defaulting firm. Logistic Regression is recommended — it catches
              the most bankruptcies (90.9% recall) even at the cost of more false alarms."),
      "investor" = list(title="Investor / Portfolio Manager View",colour="#27AE60",
        text="As an investor, PRECISION matters most. False alarms cause unnecessary divestment.
              Random Forest is recommended — fewer false alarms (73 vs 158) means less
              unnecessary portfolio disruption."),
      "regulator"= list(title="Regulator / Government View",colour="#F0C040",
        text="As a regulator, AUC-ROC and RANKING ABILITY matters. Logistic Regression's
              highest AUC (0.941) and full interpretability satisfies GDPR explainability
              requirements better than black-box models.")
    )
    tags$div(
      tags$h4(style=paste0("color:",ctx$colour),ctx$title),
      tags$p(style="color:#CCC;font-size:13px;",ctx$text)
    )
  })

  # Prediction logic
  pred_results <- eventReactive(input$predict_btn, {

    # Raw input vector
    raw_vals <- c(
      as.numeric(input$inp_f1),  as.numeric(input$inp_f2),  as.numeric(input$inp_f3),
      as.numeric(input$inp_f4),  as.numeric(input$inp_f5),  as.numeric(input$inp_f6),
      as.numeric(input$inp_f7),  as.numeric(input$inp_f8),  as.numeric(input$inp_f9),
      as.numeric(input$inp_f10), as.numeric(input$inp_f11), as.numeric(input$inp_f12),
      as.numeric(input$inp_f13), as.numeric(input$inp_f14), as.numeric(input$inp_f15),
      as.numeric(input$inp_f16), as.numeric(input$inp_f17), as.numeric(input$inp_f18),
      as.numeric(input$inp_f19), as.numeric(input$inp_f20),
      as.numeric(input$inp_liab_flag),
      as.numeric(input$inp_income_flag)
    )

    # Raw data frame (LR and RF)
    new_data_raw <- as.data.frame(matrix(raw_vals, nrow=1))
    colnames(new_data_raw) <- FEATURE_NAMES

    # Scaled data frame (NN)
    scaled_num   <- predict(SCALER, new_data_raw[, NUMERIC_FEATS, drop=FALSE])
    new_data_nn  <- as.data.frame(scaled_num)
    new_data_nn$Liability.Assets.Flag <- as.numeric(input$inp_liab_flag)
    new_data_nn$Net.Income.Flag       <- as.numeric(input$inp_income_flag)
    new_data_nn  <- new_data_nn[, FEATURE_NAMES]

    # LR — raw inputs + Bankrupt factor
    new_data_lr          <- new_data_raw
    new_data_lr$Bankrupt <- factor("No", levels=c("No","Yes"))
    lr_prob <- tryCatch({
      p <- predict(lr_results$model, newdata=new_data_lr, type="prob")
      as.numeric(p[1,"Yes"])
    }, error=function(e){ cat("LR error:",conditionMessage(e),"\n"); NA })

    # RF — raw inputs only
    rf_prob <- tryCatch({
      p <- predict(rf_results$model, newdata=new_data_raw, type="prob")
      as.numeric(p[1,"Yes"])
    }, error=function(e){ cat("RF error:",conditionMessage(e),"\n"); NA })

    # NN — scaled inputs
    # Factor encoding: 1 = bankrupt, 2 = not bankrupt
    # NN single output: high (~1) = bankrupt, low (~0) = not bankrupt
    # So raw output IS the bankruptcy probability directly
    nn_prob <- tryCatch({
      p <- predict(nn_results$model, newdata=new_data_nn)
      cat("NN raw output:", p, "\n")
      raw <- if (is.matrix(p)||is.data.frame(p)) as.numeric(p[1,1]) else as.numeric(p[1])
      raw
    }, error=function(e){ cat("NN error:",conditionMessage(e),"\n"); NA })

    cat(sprintf("PREDICTIONS — LR: %.3f | RF: %.3f | NN: %.3f\n",
                ifelse(is.na(lr_prob),-1,lr_prob),
                ifelse(is.na(rf_prob),-1,rf_prob),
                ifelse(is.na(nn_prob),-1,nn_prob)))

    list(lr=lr_prob, rf=rf_prob, nn=nn_prob)
  })

  # Probability bar
  prob_bar <- function(prob, model_name, colour) {
    if (is.na(prob)) {
      return(tags$div(style="margin:12px 0;padding:8px 0;",
        tags$span(style=paste0("color:",colour,";font-weight:bold;"),model_name),
        tags$span(style="color:#888;font-size:12px;margin-left:8px;","— could not generate prediction")
      ))
    }
    pct   <- round(prob*100,1)
    label <- if(prob>=0.5)"HIGH RISK" else "LOW RISK"
    lcol  <- if(prob>=0.5)"#E74C3C"   else "#2ECC71"
    tags$div(style="margin:12px 0;",
      tags$div(style="display:flex;justify-content:space-between;margin-bottom:4px;",
        tags$span(style=paste0("color:",colour,";font-weight:bold;"),model_name),
        tags$span(style=paste0("color:",lcol,  ";font-weight:bold;"),label)
      ),
      tags$div(style="background:#0D0D1A;border-radius:4px;height:24px;width:100%;",
        tags$div(style=paste0("background:",colour,";width:",pct,"%;height:100%;",
                              "border-radius:4px;display:flex;align-items:center;padding-left:8px;"),
          tags$span(style="color:white;font-size:12px;font-weight:bold;",paste0(pct,"%"))
        )
      )
    )
  }

  output$pred_lr <- renderUI({ req(input$predict_btn); prob_bar(pred_results()$lr,"Logistic Regression","#2980B9") })
  output$pred_rf <- renderUI({ req(input$predict_btn); prob_bar(pred_results()$rf,"Random Forest",      "#27AE60") })
  output$pred_nn <- renderUI({ req(input$predict_btn); prob_bar(pred_results()$nn,"Neural Network",     "#8E44AD") })

  # Verdict
  output$pred_verdict <- renderUI({
    req(input$predict_btn)
    res   <- pred_results()
    probs <- c(res$lr, res$rf, res$nn)
    avg   <- mean(probs, na.rm=TRUE)
    votes <- sum(probs >= 0.5, na.rm=TRUE)
    verdict <- if(votes>=2)"BANKRUPT" else "NOT BANKRUPT"
    vcol    <- if(votes>=2)"#E74C3C"  else "#2ECC71"
    vbg     <- if(votes>=2)"#3D0000"  else "#003D00"
    tags$div(style=paste0("background:",vbg,";border:2px solid ",vcol,
                          ";border-radius:8px;padding:15px;text-align:center;"),
      tags$div(style=paste0("font-size:22px;font-weight:bold;color:",vcol),paste0("VERDICT: ",verdict)),
      tags$div(style="color:#CCC;margin-top:6px;font-size:13px;",
               paste0("Models in agreement: ",votes,"/3 | Avg probability: ",round(avg*100,1),"%"))
    )
  })

  # Stakeholder advice
  output$stakeholder_advice <- renderUI({
    req(input$predict_btn)
    res   <- pred_results()
    votes <- sum(c(res$lr,res$rf,res$nn)>=0.5, na.rm=TRUE)
    advice <- switch(input$stakeholder,
      "lender"   = if(votes>=2)
        "RECOMMENDATION: Decline credit application or require significant collateral.
         Logistic Regression flags high risk — further due diligence strongly advised."
      else
        "RECOMMENDATION: Credit application appears low risk. Standard due diligence applies.
         Monitor Debt Ratio and Net Income trends going forward.",
      "investor" = if(votes>=2)
        "RECOMMENDATION: Consider divesting or reducing exposure. Random Forest flags
         elevated risk. Review position sizing and portfolio concentration."
      else
        "RECOMMENDATION: Company financials appear stable. No immediate divestment indicated.
         Continue monitoring quarterly statements for deterioration.",
      "regulator"= if(votes>=2)
        "RECOMMENDATION: Flag for enhanced regulatory scrutiny. Multiple models signal distress.
         Consider stress testing and increased reporting frequency."
      else
        "RECOMMENDATION: No immediate regulatory concern. Standard monitoring applies.
         Review annually or if sector conditions deteriorate."
    )
    acol <- if(votes>=2)"#E74C3C" else "#2ECC71"
    tags$div(
      style=paste0("border-left:4px solid ",acol,
                   ";padding:12px 15px;background:#0F3460;",
                   "border-radius:0 5px 5px 0;margin-top:5px;"),
      tags$p(style=paste0("color:",acol,";font-size:13px;margin:0;"),advice)
    )
  })

}

# =============================================================================
# RUN APP
# =============================================================================

shinyApp(ui = ui, server = server)