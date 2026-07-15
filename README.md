Step 1 (triage_model): medical / non_medical
├─ medical → Step 2 (med_model): AGENT3, unchanged → medical_clinical / medical_healthcheck / medical_lab / medical_others
└─ non_medical → Step 2 (router_model): financial / identification / eform / unrelated_document
    ├─ financial → Step 3 (financial_model): bankstatement / bookbank / companyregistration / receipt / selfincomedeclaration / others
    ├─ identification → Step 3 (id_model): the 10 existing leaves + 2 new (see caveat below)
    ├─ eform → terminal, no Step 3 call
    └─ unrelated_document → terminal, no Step 3 call