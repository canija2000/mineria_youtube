# Publicar la web en AWS (EC2 con los créditos del plan gratuito)

Una máquina virtual EC2 de 8 GB corre el **mismo contenedor que Cloud Run**, con los datos y el modelo
adentro. Caddy lo publica por los puertos 80 y 443 y saca el certificado HTTPS solo. Los videos que
llegan por la web se guardan en el disco de la máquina (`~/mapa/persistente`), así que sobreviven a
cada despliegue.

```
Mac: deploy/aws/desplegar.sh ──rsync──► EC2  ~/mapa/contexto  ──docker build──► contenedor "mapa"
internet ──► :80/:443 Caddy ──► 127.0.0.1:8000 contenedor ──► ~/mapa/persistente (catálogo + miniaturas)
```

> Esta guía no se ha probado de punta a punta en AWS. Los pasos donde la gente suele trabarse
> van marcados con ⚠️.

## Costo

| ítem | aprox. por mes encendida 24/7 (us-east-1) |
|---|---|
| `m7i-flex.large` (2 vCPU, 8 GB) | ~70 USD |
| disco de 30 GB gp3 | ~2,5 USD |
| IP pública (Elastic IP) | ~3,6 USD |
| **total** | **~76 USD** |

Si la detienes (*Stop*), solo cobra el disco y la IP (~6 USD/mes). Con el plan gratuito, AWS
**detiene la cuenta** cuando se acaban los créditos o el periodo de 6 meses: no llega una factura,
pero la web se cae. Si no pasas la cuenta a un plan pagado, después se borra lo que haya adentro.
Antes de esa fecha, respalda el catálogo (ver la sección 7).

---

## 0. Borrar la base RDS de la actividad (para que no gaste créditos)

1. En la consola, abre **RDS → Bases de datos** (*Databases*). Revisa la región, arriba a la derecha:
   tiene que ser la misma donde la creaste.
2. ⚠️ Si la base tiene **protección contra eliminación**, selecciónala y entra a **Modificar** (*Modify*).
   Desmarca *Habilitar la protección contra eliminación*, elige **Continuar** y después **Aplicar inmediatamente**.
3. Selecciónala y entra a **Acciones → Eliminar** (*Actions → Delete*):
   - desmarca **Crear una instantánea final** (*Create final snapshot*);
   - desmarca **Retener copias de seguridad automatizadas**;
   - marca el aviso de que se pierden los datos y escribe `delete me`.
4. Si es **Aurora**, se borra en dos pasos: primero la instancia (*writer*) y después el clúster.
5. Al final, revisa **RDS → Instantáneas** (*Snapshots*). Si quedó alguna, bórrala también:
   las instantáneas cobran almacenamiento.

## 1. Alerta de presupuesto (si no la creaste con la actividad de Budgets)

**Billing → Budgets → Create budget → Monthly cost budget**. Ponle un monto, por ejemplo 80 USD, y tu
correo para los avisos.

## 2. Región

Arriba a la derecha elige **US East (N. Virginia) `us-east-1`**, que suele ser la más barata.
Todo lo que sigue se crea en esa región.

## 3. Crear la máquina

**EC2 → Instancias → Lanzar instancias**.

| campo | valor |
|---|---|
| Nombre | `mapa-youtube` |
| Imagen (AMI) | **Ubuntu Server 24.04 LTS**, arquitectura **64 bits (x86)** |
| Tipo de instancia | **`m7i-flex.large`** (2 vCPU, 8 GiB). ⚠️ Con 4 GB o menos el servidor muere por falta de memoria. Si la consola no la deja elegir, busca otro tipo permitido con al menos 8 GiB; `c7i-flex.large` no sirve, porque tiene 4 |
| Par de claves | **Crear un nuevo par de claves** → nombre `aws_mapa`, tipo ED25519, formato `.pem`. Se descarga una sola vez |
| Red → Grupo de seguridad | **Crear grupo de seguridad**. Marca *Permitir SSH* con origen **Mi IP**, *Permitir HTTPS desde Internet* y *Permitir HTTP desde Internet* |
| Almacenamiento | **30 GiB gp3** (la imagen pesa 3–4 GB y se construye en la máquina) |

Elige **Lanzar instancia**. En tu Mac, mueve la llave a su lugar:

```bash
mv ~/Downloads/aws_mapa.pem ~/.ssh/ && chmod 600 ~/.ssh/aws_mapa.pem
```

## 4. IP fija (Elastic IP)

Sin este paso, la IP pública cambia cada vez que detienes y vuelves a encender la máquina.

1. Abre **EC2 → IP elásticas → Asignar la dirección IP elástica → Asignar**.
2. Selecciónala y entra a **Acciones → Asociar la dirección IP elástica**. Como instancia elige `mapa-youtube` y confirma con **Asociar**.

Anota esa IP. En lo que sigue aparece como `<IP>`.

⚠️ **"Mi IP" en el grupo de seguridad es la IP de tu casa en ese momento.** Si cambias de red,
`ssh` se queda pegado. Para arreglarlo, entra a **EC2 → Grupos de seguridad**, abre el grupo, elige
**Editar reglas de entrada**, y en la regla SSH vuelve a elegir **Mi IP**.

## 5. Desplegar (desde el Mac)

```bash
ssh -i ~/.ssh/aws_mapa.pem ubuntu@<IP> echo conectado
```

```bash
deploy/aws/desplegar.sh <IP>
```

El script, en orden:

1. Baja el catálogo de videos recibidos que haya en la máquina y lo suma al local.
2. Arma el contexto de la imagen con `deploy/cloudrun/preparar.sh`: código, datos y modelo.
3. Lo sube con `rsync`. La primera vez son ~2 GB; después, solo lo que cambió.
4. En la máquina hace lo siguiente:
   - instala lo que falte: swap de 4 GB, Docker y Caddy;
   - construye la imagen;
   - levanta el contenedor `mapa` con reinicio automático y la llave de YouTube de tu `.env`;
   - configura Caddy y espera a que `/health` responda.

Al final imprime la dirección: `http://<IP>`.

**HTTPS con un dominio gratis (recomendado):**

1. En https://www.duckdns.org crea un subdominio, por ejemplo `mapa-youtube`, que apunte a `<IP>`.
2. Vuelve a desplegar con el dominio: `deploy/aws/desplegar.sh <IP> mapa-youtube.duckdns.org`. Basta una vez: los despliegues siguientes sin dominio mantienen el HTTPS.
   Caddy pide el certificado solo, y en un minuto funciona `https://mapa-youtube.duckdns.org`.

## 6. Ciclo de actualización (igual que en Cloud Run)

```bash
deploy/aws/bajar_catalogo.sh <IP>      # trae los videos que llegaron por la web
.venv/bin/python remap.py              # mapa nuevo que los incluye
deploy/aws/desplegar.sh <IP> [dominio] # sube y reinicia
```

## 7. Mantenimiento

| tarea | comando |
|---|---|
| logs | `ssh -i ~/.ssh/aws_mapa.pem ubuntu@<IP> sudo docker logs -f mapa` |
| memoria | `ssh -i ~/.ssh/aws_mapa.pem ubuntu@<IP> sudo docker stats --no-stream mapa` |
| respaldar lo ingresado | `deploy/aws/bajar_catalogo.sh <IP>` (lo suma a `data/webapp_catalog.sqlite3`) |
| apagar para ahorrar | **EC2 → Instancias → Estado de la instancia → Detener**. Al encenderla, el contenedor y Caddy arrancan solos (~1 min en cargar el mapa) |
| borrar todo al final | **Terminar** la instancia y **liberar** la Elastic IP. Una IP sin asociar también cobra |

## 8. Si algo falla

| síntoma | revisar |
|---|---|
| `ssh` se queda pegado | la regla SSH con **Mi IP** del paso 4, la IP correcta y la llave en `~/.ssh/aws_mapa.pem` |
| `Permission denied (publickey)` | `chmod 600 ~/.ssh/aws_mapa.pem`; el usuario es `ubuntu` |
| `http://<IP>` no carga | reglas HTTP/HTTPS del grupo de seguridad; `sudo systemctl status caddy` en la máquina |
| Caddy responde 502 | el contenedor no está arriba: `sudo docker ps -a`, `sudo docker logs mapa` |
| el contenedor se reinicia en bucle | `sudo docker logs mapa`; con `free -h` revisa que sea la máquina de 8 GB |
| HTTPS no funciona | que el dominio apunte a la IP (`ping mapa-youtube.duckdns.org`) y el puerto 443 abierto |

## Seguridad básica

- Solo quedan abiertos 22 (solo desde tu IP), 80 y 443. El servidor de Python escucha en `127.0.0.1`.
- La llave de YouTube queda en `~/mapa/.env` de la máquina, con permisos 600.
- Si la web es pública, cualquiera puede gastar tu cuota de YouTube agregando videos o perfiles.
