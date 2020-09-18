---
title: "News"
layout: textlay
excerpt: "Deep Learning at Centro de Informática - UFPE."
sitemap: false
permalink: /allnews.html
---

# News

{% for article in site.data.news %}
<p>{{ article.date }} <br>
<em>{{ article.headline }}</em></p>
{% endfor %}
