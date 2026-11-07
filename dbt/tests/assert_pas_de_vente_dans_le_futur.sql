select *
from {{ ref('silver_ventes') }}
where date_mutation > current_date
