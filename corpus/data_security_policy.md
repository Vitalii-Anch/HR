# Northwind Retail Co. — Data Security Policy

**Doc ID:** data-security-policy
**Effective Date:** January 1, 2026
**Owner:** IT Security Team
**Applies To:** All employees, contractors, and vendors with access to Northwind systems or data.

## 1. Purpose

This policy establishes minimum security requirements for handling Company, customer, and employee data, and is a companion to the Remote Work Policy (`remote-work-policy`) and Equipment Policy (`equipment-policy`).

## 2. Data Classification

Northwind classifies data into three tiers:

- **Public:** Marketing materials, public job postings — no special handling required.
- **Internal:** Internal memos, non-sensitive operational data — should not be shared outside the Company without approval.
- **Confidential:** Customer personal data (names, payment info, addresses), employee personal data (SSNs, salary, health/leave information), financial statements pre-release, and source code. Confidential data requires encryption at rest and in transit, and access is restricted on a need-to-know basis.

## 3. Device and Access Requirements

- All employees accessing internal systems remotely must connect via the Company-approved VPN.
- Only Company-issued or Company-approved (via IT exception) devices may access Confidential data. Personal, unmanaged devices may not store Confidential data.
- Multi-factor authentication (MFA) is required for all Company accounts, including email, HR systems, and the code repository.
- Screens must be locked when unattended, and devices must use full-disk encryption (enabled by default on Company-issued laptops).

## 4. Handling Customer and Employee Personal Data

- Customer payment data must never be stored outside of Northwind's PCI-compliant payment systems.
- Employee personal data (including PTO balances, leave records, and benefits elections) may only be accessed by HR, Payroll, and the employee's direct manager on a need-to-know basis, and must not be exported to personal devices or personal cloud storage.
- Any mock or test data used in engineering or training environments (including this course project's `mock_data/`) must be clearly synthetic and must never contain real employee or customer information.

## 5. Incident Reporting

Suspected data security incidents (lost device, phishing click, suspected unauthorized access, accidental data exposure) must be reported to IT Security within **24 hours** of discovery via the IT Security incident channel or ticketing system. Delayed reporting can worsen exposure and is itself treated as a policy concern.

## 6. Third-Party and Vendor Data Sharing

Sharing Confidential data with a vendor or third party requires a signed Data Processing Agreement (DPA) and sign-off from IT Security and Legal. Employees may not use unapproved third-party AI tools, browser extensions, or SaaS products to process Confidential data.

## 7. Remote and Home Network Security

Employees working remotely must ensure their home network uses a password-protected, encrypted Wi-Fi connection (WPA2 or better) and must not use public, unsecured Wi-Fi to access Confidential data without VPN protection.

## 8. Consequences of Violation

Violations of this policy are addressed under the Workplace Conduct Policy (`workplace-conduct-policy`) and may range from a documented warning to termination, depending on severity and whether the violation was negligent or willful.

## 9. Related Policies

- Remote Work Policy (`remote-work-policy`)
- Equipment Policy (`equipment-policy`)
- Workplace Conduct Policy (`workplace-conduct-policy`)
