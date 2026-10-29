# Logbook

Diario del laboratorio, in italiano: un file al mese (`AAAA-MM.md`), il giorno più recente in alto.
Non è un report: righe brevi, numeri e figure, anche quello che non va.

Una voce per giorno di lavoro:

```markdown
## 2026-10-29 gio · gruppi

- **Fatto:** …
- **Trovato:** … (una tabella o una figura se serve)
- **Non va:** …
- **Prossimo:** …
```

Solo le righe che hanno contenuto. Dopo `·` i temi (gruppi, fasi, viewer, slides, lab).
L'ultima riga, **Rif.:**, dà i commit del giorno.

Regole:

- Una voce passata non si riscrive. Se un risultato cambia, si aggiunge nel giorno in cui lo si
  scopre una riga **Correzione:** che dice cosa valeva prima e perché non vale più.
- La voce si legge da sola: il numero e la figura chiave stanno nella voce.
- Un report o una nota si cita con il commit (`group_vs_solo @ f57741c`), mai da solo: il file
  cambia, il commit no.

Le figure si congelano in `fig/`, perché `out/` viene sovrascritto:

```bash
logbook/fig.sh analysis/.../out/<run>/tamsd.pdf tamsd    # stampa la riga ![](fig/AAAA-MM-GG_tamsd.png)
```
