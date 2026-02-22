# ORDEN DE EJECUCIÓN:

## 0. Obtener un corpus limpio
python clean_corpus.py <br>
Renombrar archivo a **corpus_completo.txt** o modificar script para el que desees.

## 1. Entrenamiento de tokens
python train_tokenizer.py --corpus CORPUS.txt --out tokenizer.model --vocab 8192<br>
(Este genera el archivo para el argumento de --sp en el script principal. **vocab=8192** opcional como parámetro)

## 2. Pretokenizado
python pretokenize.py --corpus CORPUS.txt --sp tokenizer.model --out tokens.pt<br>
(Este genera el archivo para --tokens en el script principal, **--out=tokens.pt** opcional como parámetro)

## 3. Entrenamiento
python train.py --corpus CORPUS.txt --sp tokenizer.model --tokens tokens.pt --epochs 3 --out tinyllama_wiki<br>
 * **--corpus** obligatorio
 * **--sp** obligatorio
 * **--tokens** opcional, por defecto "tokens.pt"
 * **--out** opcional, por defecto "tinyllama_wiki"
 * **--epochs** opcional pero necesario, por defecto "3"
 * **--resume** para continuar entrenamiento a partir de cierta época, indicar ruta
 * **--batch** opcional, por defecto "8"
 * **--accum_steps** opcional, por defecto "8"
 * **--lr** tasa de aprendizaje, opcional, por defecto "2e-4"
 * **--stride** pasos, opcional, por defecto "256"
 * **--max_tokens** opcional, por defecto "12_000_000" (12M)

## 3.1 Para continuar entrenamiento:
python train.py --corpus corpus_completo.txt --sp tokenizer.model --tokens wiki_tokens.pt --out tinyllama_wiki --resume tinyllama_wiki/checkpoint10.pt --epochs 20<br>
 
## 4. Conversión a HF (Hugging Face)
python export_hf_wiki.py --ckpt checkpoint.pt --tokenizer tokenizer.model --out tinyllama_hf<br>
(--out opcional; este script genera config.json, model.safetensors y tokenizer.model (ya existente))

## 5. Conversión a GGUF
python convert-hf-to-gguf.py tinyllama_hf --outfile tinyllama.gguf --outtype f16

---
## Preguntas frecuentes: 

**¿Porqué usar este repositorio / para que?** En un principio, fué planeado como hobby/estudio. Quería comprobar la factibilidad de crear uno propio en la comodidad de mi hogar así como validar si con una PC de escritorio de prestaciones medias, era posible. En mi entorno de trabajo, requiero de asistencia especializada para agilizar mis procesos de desarrollo y consulta de materiales bibliográficos, de ahí que requiriera de una fuente de consulta con cierta capacidad cognitiva. Como no siempre puedo contar con fluído eléctrico, acceso a Internet y referencias especializadas, opté por implementar un LLM pequeño que pudiera ejecutar en casi cualquier PC sin que ocupe recursos cuantiosos y estuviera especializado en la temática de mi interés particular. Este proyecto servirá de base para ello y como prototipo, mi objetivo es cumplir con cada aspecto mencionado, una vez logrado, generalizar el proceso para expandirlo a cualquier área.
**¿Tiene posibilidades de mejora?** Absolutamente, tiene todas las características para convertirse en una herramienta para el desarrollo a nivel profesional si recibiera la ayuda y apoyo de especialistas en el campo.
**¿Cómo puede fallar?** si Config no coincide con la arquitectura real, vocab_size difiere de tokenizer.model y shapes no coinciden.
**¿Qué tipo de modelo obtendré?** Para wikipedia 2026 (eswiki-latest-pages-articles.xml.bz2), se obtiene un modelo con cierta coherencia a partir de las 3 primeras épocas, con ajustes en los parámetros y según tu equipo de cómputo, podrías tener un modelo útil pero básico y muy estilo enciclopédico a partir de las 10 épocas. Cada checkpoint ocupa cerca de 336MB y su modelo para inferencia 112MB, el modelo convertido a GGUF sólo 56.2MB.
**¿Qué requisitos de PC requiero?** Este conjunto de códigos ha sido probado en un Intel Xeon E5-2650 V4(12 núcleos y 24 hilos), 96GB de RAM y NVIDIA GeForce 1060 6GB, 512GB de almacenamiento NVME dedicados, usando CUDA 118; logrando que cada época tarde sólo 30min. 
