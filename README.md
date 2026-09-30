# neonatal_icu_model
Script file to fit ML models for predicting ventilatory outcomes in 
a neonatal ICU dataset. 

This code describes the primary analysis for the paper: 
Computer Vision–Enabled Early Prediction of Neonatal Respiratory Escalation 
in Resource-Limited Settings: An Indian Multi-Center Cohort Study.

Primary code was written by Amrit Sharma, supervised by Prof Brody H Foy.

Contact: Brody H Foy, DPhil. brodyfoy@uw.edu

The core file uses a neonatal ICU dataset (cannot be shared due to PHI restrictions) 
and fits one of three model types (random_forest, decision_tree, logistic_regression) 
to predict outcomes (mortality, escalation of ventilatory support, future requirement of 
intubation or non-invasive ventilation [NIV]) at one of three timepoints (6h, 12h or 
24h post birth), using demographics and vitals measurements.

Example use
--------
python Fit_models.py \
    --timepoint 6 --outcome mortality --model random_forest

Notes
----------------------------------------------------
* Transfers to another ICU or a higher centre of care are excluded, due to indeterminate
outcomes.
* Discharge against medical advice (DAMA) is not excluded. DAMA is a
  non-death for the mortality target, while respiratory outcomes use the
  patient's observed subsequent ventilation records up to time of dicharge.
* Escalation excludes patients already invasively ventilated at time of prediction
(since by definition they cannot escalate). Similarly, NIV/intubation outcome 
excludes patients already receiving
  NIV or invasive ventilation.
