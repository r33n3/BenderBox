# Finding: {{ title }}

**Severity:** {{ severity }}
**Category:** {{ category }}
**Status:** {{ status }}

## Description

{{ description }}

{% if location %}
## Location

{{ location }}
{% endif %}

{% if code_snippet %}
## Code Reference

```
{{ code_snippet }}
```
{% endif %}

{% if recommendation %}
## Recommendation

{{ recommendation }}
{% endif %}

{% if cwe_id %}
## References

- CWE: {{ cwe_id }}
{% endif %}
