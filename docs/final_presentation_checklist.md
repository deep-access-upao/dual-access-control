# Checklist final de presentación

## 1. Antes de la exposición

- [ ] Trabajar en Windows nativo desde la raíz del repositorio.
- [ ] Activar `.venv` y comprobar que las dependencias estén instaladas.
- [ ] Confirmar que el modelo privado existe sin mostrar ni copiar el `.keras`.
- [ ] Ejecutar `check-model`, tests y `git status`.
- [ ] Iniciar la web y verificar `/health`.
- [ ] Usar un UID simulado y nombre ficticio.
- [ ] Preparar referencias/capturas con consentimiento y fuera del repositorio.
- [ ] Cerrar notificaciones y ocultar rutas, pestañas y datos personales.
- [ ] Probar cámara; mantener carga de archivo como alternativa.
- [ ] Tener una segunda terminal lista para comandos.

## 2. Capturas que se deben tomar

- [ ] Portada del proyecto y arquitectura general.
- [ ] Tabla del dataset: 11 personas, 44 videos, 1,544 utilizables y splits.
- [ ] Auditoría con cero fuga en imágenes, videos, pares y hashes.
- [ ] Configuración de entrenamiento sin rutas privadas.
- [ ] Tabla de test limpio final y matriz de confusión.
- [ ] Curva ROC y curva precision-recall existentes.
- [ ] Comparación con/sin augmentation.
- [ ] Comparación `max_f1` vs `security_first`.
- [ ] Stress tests destacando poca luz, sobreexposición y bajo contraste.
- [ ] Comparación baseline vs GAP + L2 + coseno y decisión final.
- [ ] Inicio web con `RFID simulado + verificación facial`.
- [ ] Registro con nombre ficticio y UID `01020304`.
- [ ] `GRANTED / RFID_AND_FACE_OK` con score y threshold visibles.
- [ ] `DENIED / RFID_UNKNOWN`.
- [ ] `DENIED / FACE_NO_MATCH` con score bajo threshold.
- [ ] `DENIED / USER_INACTIVE`.
- [ ] Historial con los cuatro motivos/decisiones.
- [ ] Terminal con `check-model` correcto.
- [ ] Terminal con suite de tests en `OK`.
- [ ] Diagrama `UID -> usuario -> rostro -> decisión -> historial`.

No publicar fotos faciales. Cuando una captura de la web deba demostrar el flujo, recortar o anonimizar cualquier imagen y conservar la evidencia únicamente en el medio autorizado para la exposición.

## 3. Comandos listos

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.inference.cli check-model --config config/model_config.json
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m src.web.app
python scripts/run_web_demo_smoke_test.py --health-only
```

Inferencia opcional con rutas privadas preparadas, sin escribirlas en diapositivas públicas:

```powershell
python -m src.inference.cli verify-pair `
  --reference "C:\ruta\referencia.jpg" `
  --capture "C:\ruta\captura.jpg"
```

## 4. Archivos que conviene mostrar

- [ ] `README.md`: visión completa y comandos principales.
- [ ] `docs/final_methodology.md`: protocolo y decisiones experimentales.
- [ ] `notebooks/99_project_summary.ipynb`: hilo de la exposición.
- [ ] `config/model_config.json`: modelo, threshold y regla.
- [ ] `src/models/siamese_network.py`: pesos compartidos y distancia L1.
- [ ] `src/models/siamese_variants.py`: GAP + L2 + coseno.
- [ ] `src/inference/`: modularidad de inferencia.
- [ ] `src/web/services.py`: aplicación de la regla dual.
- [ ] `docs/integrated_demo_test_plan.md`: casos de extremo a extremo.
- [ ] `docs/rubric_coverage.md`: correspondencia con los 20 criterios.

## 5. Evidencia de métricas

- [ ] Test final: accuracy 0.9660, precision 0.9794, recall 0.9520, F1 0.9655.
- [ ] Seguridad: FAR 0.0200, FRR 0.0480, ROC AUC 0.9838.
- [ ] Matriz: TN 245, FP 5, FN 12, TP 238.
- [ ] Threshold final: `0.3128704727`, calibrado solo con validation.
- [ ] Explicar el costo: frente a `max_f1`, FP baja 9 -> 5 y FN sube 1 -> 12.
- [ ] Stress: poca luz FRR 0.356, sobreexposición 0.320, bajo contraste 0.304.
- [ ] GAP: mejor limpio, pero FRR 1.00/0.90/0.82 en las tres condiciones fotométricas.

Gráficos versionados recomendados:

```text
outputs/experiments/baseline_formal/baseline_con_aumento/test_confusion_matrix.png
outputs/experiments/baseline_formal/baseline_con_aumento/test_clean_roc_curve.png
outputs/experiments/baseline_formal/baseline_con_aumento/test_clean_precision_recall_curve.png
outputs/experiments/model_comparison/clean_test_comparison.png
outputs/experiments/model_comparison/photometric_frr_comparison.png
```

## 6. Puntos que se deben explicar oralmente

- [ ] Por qué verificar no es lo mismo que clasificar identidades.
- [ ] Por qué Deep Learning es adecuado para variaciones visuales complejas.
- [ ] Por qué el split aleatorio de pares causaba fuga.
- [ ] Qué mide el split por video y qué no mide.
- [ ] Por qué augmentation solo pertenece a train.
- [ ] Cómo se eligió `security_first` sin mirar test.
- [ ] Diferencia entre FAR —intrusión— y FRR —rechazo legítimo—.
- [ ] Por qué el baseline aumentado quedó como modelo final.
- [ ] Por qué GAP + L2 + coseno fue descartado aunque mejora test limpio.
- [ ] Por qué no se implementaron fine-tuning ni distillation en este alcance.
- [ ] Cómo la inferencia modular evita thresholds duplicados.
- [ ] Por qué RFID es simulado y qué interfaz sustituiría un lector físico.
- [ ] Limitaciones: 11 personas, sin externos, stress sintético y mala iluminación.
- [ ] Privacidad, consentimiento, sesgos y alcance académico.

## 7. Guion breve de demo

1. Mostrar `/health` y el threshold.
2. Registrar usuario ficticio y referencias consentidas.
3. Ejecutar caso positivo y señalar que ambos factores son necesarios.
4. Ejecutar UID desconocido para demostrar que no se invoca el modelo.
5. Ejecutar rostro incorrecto y usuario inactivo.
6. Mostrar historial con score, threshold, decisión y motivo.
7. Cerrar con la matriz de confusión y las limitaciones fotométricas.

## 8. Advertencias y plan de contingencia

- [ ] No bajar el threshold si una captura falla; recapturar con mejor luz.
- [ ] No recalibrar ni entrenar durante la presentación.
- [ ] No depender solo de cámara: tener archivos locales autorizados.
- [ ] No reutilizar UID existente en el smoke test.
- [ ] Si el modelo no carga, mostrar `check-model`, config y evidencias ya capturadas; no improvisar otro modelo.
- [ ] Si la web no inicia, comprobar puerto 8000 y dependencias; conservar capturas anonimizadas de respaldo.
- [ ] Si no hay Internet, la demo local debe seguir funcionando.
- [ ] No abrir carpetas de dataset, support set, base SQLite o modelo ante el público.
- [ ] No afirmar que el sistema está listo para producción.

## 9. Revisión previa a entrega

- [ ] Notebook sin outputs pesados ni fotos incrustadas.
- [ ] README enlaza metodología, uso, rúbrica y checklist.
- [ ] Cifras coinciden con `config/model_config.json` y outputs.
- [ ] No hay `.keras`, SQLite, imágenes, videos, datasets, checkpoints o cachés en el diff.
- [ ] El commit contiene solo documentación y el notebook resumen.
- [ ] La matriz cubre los 20 ítems y marca fine-tuning/distillation como no implementados.
