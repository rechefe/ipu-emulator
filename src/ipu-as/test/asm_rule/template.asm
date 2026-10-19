{#- ipu:params
cpr: int = 1
-#}
{% for i in range(cpr) %}SET LR{{ i }} CR2;;
{% endfor %}{% include "body.asm.j2" %}
