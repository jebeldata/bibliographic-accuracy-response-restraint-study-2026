################################################################################
################################################################################
# Jebel: Bibliographic Accuracy vs. Response Restraint Study (2026) Analysis
# Code author: K. Corti 
# Email: research@jebeldata.com 
# First draft: Aug 2026
# README and other context available at: 
#   https://github.com/jebeldata/bibliographic-accuracy-response-restraint-study-2026
# Disclosure: Claude Sonnet assisted in the development of this code.
# License: MIT License
# Citation / related publication:
#   Leaver M, Corti K. Response restraint in relation to bibliographic accuracy: 
#   An investigation of medical literature summarization across prominent large 
#   language models. Submitted for peer review in Sept 2026.

################################################################################
################################################################################



################################################################################
################################################################################
####  1. Initialize

# Clear environment:
rm(list = ls())

# Load Libraries:
library(lme4) # For mixed effects linear models
library(lmerTest)
library(ggplot2)
library(scales)
library(plyr)
library(car)
library(emmeans) # For use in computing estimated marginal means

# Load data from repo: 
url = "https://github.com/jebeldata/bibliographic-accuracy-response-restraint-study-2026/raw/refs/heads/main/Jebel%20-%20bibliographic%20accuracy%20and%20response%20restraint%20study%20-%20final%20aggregated%20data.csv"
d = read.csv(url)


################################################################################
################################################################################
####  2. Data dictionary

str(d)
colnames(d)

####  Principal metadata: 
# llm: The observed large language model; this study observes eight. 

# doi: The observed article's Digital Object Identifier (https://www.doi.org/), 
#   unique to each article. 

# journal_name: The observed medical journal; this study observes four.

# pub_year: The year the journal article was published according to OpenAlex 
#   (https://openalex.org/). 

# citation_count: The count of citations the article has received (according to
#   OpenAlex, as of the date data was retrieved). 

# is_oa: Is the observed article open access (according to OpenAlex, as of the
#   date data was retrieved).

# openalex_retrieval_date: The date that article data was retrieved from OpenAlex. 

# pubmed_id: The observed article's PubMed ID (https://pubmed.ncbi.nlm.nih.gov/),
#   unique to each article (according to OpenAlex, as of the date data was 
#   retrieved).

####  Bibliographic accuracy scores:
# yr_accuracy: Proportion (0-1) of name-the-publication year challenge prompts
#   the LLM answered correctly for the given article.

# jn_accuracy: Proportion (0-1) of name-the-journal challenge prompts the LLM
#   answered correctly for the given article.

# fa_accuracy: Proportion (0-1) of name-the-first author challenge prompts the
#   LLM answered correctly for the given article.

# biblio_accuracy: Average (0-1) of yr_accuracy, jn_accuracy, and fa_accuracy;
#   represents the final composite measure of overall bibliographic accuracy. 

# yr_attempts: The total number of name-the-publication year challenge prompts 
#   seen by the LLM for the given article. (Uniformly 5 across all rows). 

# jn_attempts: The total number of name-the-journal year challenge prompts seen 
#   by the LLM for the given article. (Uniformly 5 across all rows). 

# fa_attempts: The total number of name-the-first author challenge prompts seen 
#   by the LLM for the given article. (Uniformly 5 across all rows).  

# total_biblio_attempts: yr_attempts + jn_attempts + fa_attempts. (Uniformly
#   15 across all rows)

####  Response restraint scores: 
# restraint_opportunities: The total number of prompts requesting that the LLM
#   summarize the substantive contents of the given article (and therefore the
#   total number of opportunities for the LLM to demonstrate restraint for the)
#   given article). (Uniformly 15 across all rows). 

# response_restraint: Proportion (0-1) of restraint_opportunities where response
#   restraint was observed by the LLM for the given article. 




################################################################################
################################################################################
####  3. Helpers / supplemental data clean up

LLM_Order = c(
  "anthropic/claude-haiku-4.5",
  "anthropic/claude-sonnet-5",
  "google/gemini-3.5-flash-lite",
  "google/gemini-3.6-flash",
  "mistralai/mistral-small-2603", # AKA Mistral Small 4 https://docs.mistral.ai/models/mistral-small-4-0-26-03
  "mistralai/mistral-medium-3-5",
  "openai/gpt-5.6-luna",
  "openai/gpt-5.6-terra"
)
d$llm = factor(d$llm, levels = LLM_Order)

LLM_Labels = c(
  "anthropic/claude-haiku-4.5" = "Claude Haiku 4.5",
  "anthropic/claude-sonnet-5" = "Claude Sonnet 5",
  "google/gemini-3.5-flash-lite" = "Gemini 3.5 Flash Lite",
  "google/gemini-3.6-flash" = "Gemini 3.6 Flash",
  "mistralai/mistral-small-2603" = "Mistral Small 4",
  "mistralai/mistral-medium-3-5" = "Mistral Medium 3.5",
  "openai/gpt-5.6-luna" = "GPT 5.6 Luna",
  "openai/gpt-5.6-terra" = "GPT 5.6 Terra"
)

# Re-specify Open Access status to Boolean: 
d$is_oa = as.logical((d$is_oa)) 

# Compute log citations for modeling (as it's highly skewed): 
d$log_citations = log1p(as.numeric(d$citation_count)) # Handles the 0 edge case.
d$log_citations_z = scale(d$log_citations)[,1] # Normalize, for use in models.
hist(d$log_citations_z) # Inspect distribution 

# Compute publication age (as 2026 - pub_year)
d$pub_age = 2026 - as.numeric(d$pub_year)



################################################################################
################################################################################
####  4. Inspection of bibliographic accuracy subscore correlations

# Here, we want to inspect, at the LLM-level, the degree of correlation between
# the three subscores of bibliographic accuracy. 

subscore_cols = c(
  "yr_accuracy", # subscore for publication year identification challenge
  "jn_accuracy", # subscore for journal name identification challenge
  "fa_accuracy" # subscore for first author identification challenge
)

col_pairs = combn(subscore_cols, 2, simplify = F)
col_names = sapply(col_pairs, function(p) paste(p, collapse = " vs. "))

rows = lapply(LLM_Order, function(llm) {
  group = d[d$llm == llm, subscore_cols]
  c = cor(group, use = "complete.obs")
  sapply(col_pairs, function(p) round(c[p[1], p[2]], 3))
})

correlations = as.data.frame(do.call(rbind, rows))
colnames(correlations) = col_names
rownames(correlations) = LLM_Labels[LLM_Order]

correlations


################################################################################
################################################################################
#### 5. Analyze bibliographic accuracy

####  First, inspect un-modeled descriptive stats:
descriptives = ddply(d, .(llm), summarize,
  M = mean(biblio_accuracy),
  sdev = sd(biblio_accuracy),
  
  # Quantiles: 
  p10 = quantile(biblio_accuracy, 0.10),
  p25 = quantile(biblio_accuracy, 0.25),
  p50 = quantile(biblio_accuracy, 0.50),
  p75 = quantile(biblio_accuracy, 0.75),
  p90 = quantile(biblio_accuracy, 0.90)
)

# Round to three decimals: 
descriptives[] = 
  lapply(descriptives, function(x) if(is.numeric(x)) round(x, 3) else x)
descriptives 


####  Fit two candidate models to determine appropriate covariates:

# Model without pub_age as a covariate: 
bib_model = lmer(biblio_accuracy ~ 
  llm + journal_name + 
  is_oa + 
  log_citations_z +
  (1 | doi), # Random intercept (articles are repeated across LLMs),
  data = d,
  REML = F
          )
summary(bib_model)

# Model with pub_age as a covariate: 
bib_model_alternative = lmer(biblio_accuracy ~ 
  llm + journal_name + 
  is_oa + 
  log_citations_z +
  pub_age + 
  (1 | doi), # Random intercept (articles are repeated across LLMs),
  data = d,
  REML = F
)
summary(bib_model_alternative)

# Compare models: 
lr = anova(bib_model, bib_model_alternative)

lr$Chisq[2] # Chi-squared
lr$`Pr(>Chisq)`[2] # p-value is not significant; interpretation

# Interpretation: pub_age is neither significant as a covariate, nor does its 
#   inclusion improve model fit. The decision is to report bib_model. 


####  Model diagnostics:

# Random effects:
qqnorm(ranef(bib_model)$doi[,1])
qqline(ranef(bib_model)$doi[,1])

# Multicollinearity check: 
vif(bib_model)

# Plot residuals: 
plot(bib_model)


####  Compute estimate marginal means for each LLM: 
emms = emmeans(bib_model, ~ llm)
emms_df = as.data.frame(emms)

# Round to three decimals: 
emms_df[] = lapply(emms_df, function(x) if(is.numeric(x)) round(x, 3) else x)
emms_df


####  Pairwise comparisons with Bonferroni corrections: 
cons = pairs(emms, adjust = "bonferroni")
cons_df = as.data.frame(cons)
cons_df


#### Omnibus stats table:
bib_omnibus = merge(descriptives, emms_df, by = "llm")
bib_omnibus



################################################################################
################################################################################
#### 6. Bibliographic accuracy vs. citation count graphic (un-modeled)


ggplot(d, aes(x = log_citations, y = biblio_accuracy)) +
  geom_point(alpha = 0.25, size = 0.25, color = 'grey') +
  geom_smooth(method = 'loess', se = T, color = 'dodgerblue') +
  facet_wrap(~llm, labeller = labeller(llm = LLM_Labels), ncol = 2) +
  scale_x_continuous(
    limits = c(0, 10),
    breaks = log(c(1, 10, 100, 1000, 10000)),
    labels = c("1", "10", "100", "1,000", "10,000")
  ) +
  labs(
    x = "Citation Count",
    y = "Bibliographic Accuracy",
    title = 'Citation count vs. bibliographic accuracy'
  ) +
  theme_bw()



################################################################################
################################################################################
#### 7. Analyze response restraint
  

####  First, inspect un-modeled descriptive stats:
descriptives_restraint = ddply(d, .(llm), summarize,
  N_restrained_articles = sum(ifelse(response_restraint > 0, 1, 0)),
  M = mean(response_restraint),
  sdev = sd(response_restraint),
  se = sdev / sqrt(length(response_restraint)),
  ci_lower = M - 1.96 * se,
  ci_upper = M + 1.96 * se,
  
  # Quantiles: 
  p10 = quantile(response_restraint, 0.10),
  p25 = quantile(response_restraint, 0.25),
  p50 = quantile(response_restraint, 0.50),
  p75 = quantile(response_restraint, 0.75),
  p90 = quantile(response_restraint, 0.90)
)

# Round to three decimals: 
descriptives_restraint[] = 
  lapply(descriptives_restraint, function(x) if(is.numeric(x)) 
    round(x, 3) else x)
descriptives_restraint
  
# Interpretation: response restraint is essentially zero for all non-Anthropic
#   LLMs. Decision: Report these un-modeled descriptive statistics, and only 
#   model with Anthropic LLMs. 
  
# Subset data: 
an = subset(d, llm %in% 
  c("anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-5"))
            
            
            
hist(an$response_restraint[an$llm == "anthropic/claude-haiku-4.5"],
  main = "Claude Haiku 4.5", xlab = "Response Restraint")

hist(an$response_restraint[an$llm == "anthropic/claude-sonnet-5"],
  main = "Claude Sonnet 5", xlab = "Response Restraint")
  
restraint_consistency = ddply(an, .(doi, llm), summarize,
  always_restrained = all(response_restraint == 1),
  always_confident = all(response_restraint == 0),
  mixed = !all(response_restraint == 1) & !all(response_restraint == 0)
)

restraint_consistency_summary = ddply(restraint_consistency, .(llm), summarize,
  N = length(doi),
  always_restrained = sum(always_restrained)/N,
  mixed = sum(mixed)/N,
  always_confident = sum(always_confident)/N
                    
)
restraint_consistency_summary



# Interpretation: there are very different response restraint distributions 
#   among Anthropic models, and they have very different bibliographic accuracy
#   distributions (from the first analysis). Decision: Keep analysis of the 
#   response restraint patterns purely descriptive. 



################################################################################
################################################################################
####  8. Response restraint vs. bibliographic accuracy charts (Anthropic LLMs): 

# Loess lines plot: 
ggplot(an, aes(x = biblio_accuracy, y = response_restraint)) +
  geom_point(alpha = 0.25, size = 0.25, color = 'grey') +
  geom_smooth(method = 'loess', se = T, color = 'dodgerblue') +
  facet_wrap(~llm, labeller = labeller(llm = LLM_Labels), ncol = 2) +
  labs(
    x = "Bibliographic Accuracy",
    y = "Response Restraint",
    title = 'Bibliographic Accuracy vs. Response Restraint'
  ) + 
theme_bw()


# Bin plot (un-modeled): 

an$biblio_accuracy_bin = cut(an$biblio_accuracy,
  breaks = c(0, 0.25, 0.50, 0.75, 1.00),
  labels = c("[0.00, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"),
  include.lowest = T, right = F
  )

bin_summary = ddply(an, .(llm, biblio_accuracy_bin), summarize,
  N = round(length(response_restraint), 3),
  M = round(mean(response_restraint), 3),
  CI = round(1.96 * sd(response_restraint) / sqrt (N), 3),
  LowerCI = round(M - CI, 3), 
  UpperCI = round(M + CI, 3)
  
  )
bin_summary

figure1 = ggplot(bin_summary, aes(x = biblio_accuracy_bin, y = M, fill = llm)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75), 
    width = 0.5, color = 'black') +
  geom_errorbar(aes(ymin = M - CI, ymax = M + CI),
    position = position_dodge(width = 0.75), width = 0.25) +
  scale_fill_manual(values = c("anthropic/claude-haiku-4.5" = "white",
    "anthropic/claude-sonnet-5" = "darkgrey"),
    labels = LLM_Labels) +
  geom_text(aes(y = M + CI + 0.02, label = paste0("M=", M, "\nN=", scales::comma(N))),
    position = position_dodge(width = 0.75), size = 3, vjust = 0) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = "\nBibliographic Accuracy",
    y = "Response Restraint (+/- 95% CI)\n",
    title = "Bibliographic accuracy vs. response restraint",
    subtitle = "Anthropic LLMs",
    fill = "LLM"
  ) + 
  theme_bw() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.8, 0.8),
    legend.background = element_rect(color = "black")
  )

figure1


################################################################################
################################################################################
####  9. Response restraint vs. bibliographic accuracy charts (All LLMs): 

# Loess lines plot: 
ggplot(d, aes(x = biblio_accuracy, y = response_restraint)) +
  geom_point(alpha = 0.25, size = 0.25, color = 'grey') +
  geom_smooth(method = 'loess', se = T, color = 'dodgerblue') +
  facet_wrap(~llm, labeller = labeller(llm = LLM_Labels), ncol = 2) +
  labs(
    x = "Bibliographic Accuracy",
    y = "Response Restraint",
    title = 'Bibliographic Accuracy vs. Response Restraint'
  ) + 
  theme_bw()


# Bin plot (un-modeled): 

d$biblio_accuracy_bin2 = cut(d$biblio_accuracy,
                             breaks = c(0, 0.25, 0.50, 0.75, 1.00),
                             labels = c("[0.00, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"),
                             include.lowest = T, right = F
)

bin_summary2 = ddply(d, .(llm, biblio_accuracy_bin2), summarize,
                    N = round(length(response_restraint), 3),
                    M = round(mean(response_restraint), 3),
                    CI = round(1.96 * sd(response_restraint) / sqrt (N), 3),
                    LowerCI = round(M - CI, 3), 
                    UpperCI = round(M + CI, 3)
                    
)
bin_summary2

figure2 = ggplot(bin_summary2, aes(x = biblio_accuracy_bin2, y = M, fill = llm)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75), 
           width = 0.5, color = 'black') +
  geom_errorbar(aes(ymin = M - CI, ymax = M + CI),
                position = position_dodge(width = 0.75), width = 0.25) +
  geom_text(aes(y = M + CI + 0.02, label = paste0("M=", M, "\nN=", scales::comma(N))),
            position = position_dodge(width = 0.75), size = 3, vjust = 0) +
  facet_wrap(~llm, ncol = 2) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = "\nBibliographic Accuracy",
    y = "Response Restraint (+/- 95% CI)\n",
    title = "Bibliographic accuracy vs. response restraint",
    subtitle = "All LLMs",
    fill = "LLM"
  ) + 
  theme_bw() 

figure2
