{#- Transforme une liste Jinja en liste SQL : ['75', '92'] -> '75', '92' -#}
{% macro liste_sql(valeurs) -%}
    {%- for v in valeurs -%}'{{ v }}'{%- if not loop.last -%}, {% endif -%}{%- endfor -%}
{%- endmacro %}
