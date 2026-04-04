"""
src/monitoring/prediction_drift.py

Detects prediction drift by querying Cloud SQL for trends in
predicted RUL values over the last 4 weeks.

If the current week's median RUL has dropped more than 20% below
the 4-week rolling average, prediction drift is flagged.

This is called from entrypoint/retrain.py before deciding to retrain.
"""

import os
import logging


def check_prediction_drift() -> dict:
    """
    Query Cloud SQL for weekly median predicted RUL over the last 4 weeks.
    Flags drift if current week median has dropped > 20% below the 4-week average.

    Returns:
        {
            "prediction_drift": bool,
            "current_median":   float or None,
            "4_week_avg":       float or None,
            "drop_pct":         float or None,
            "message":          str
        }
    """
    result = {
        'prediction_drift': False,
        'current_median':   None,
        '4_week_avg':       None,
        'drop_pct':         None,
        'message':          'OK'
    }

    db_host    = os.environ.get('DB_HOST', '127.0.0.1')
    db_name    = os.environ.get('DB_NAME', 'apu_predictions')
    db_user    = os.environ.get('DB_USER', 'apu_user')
    project_id = os.environ.get('GCP_PROJECT_ID', '')

    if not project_id:
        result['message'] = 'GCP_PROJECT_ID not set — skipping prediction drift check.'
        logging.info(result['message'])
        return result

    try:
        import psycopg2
        from google.cloud import secretmanager

        sm_client = secretmanager.SecretManagerServiceClient()
        secret_name = f'projects/{project_id}/secrets/db-password/versions/latest'
        password = sm_client.access_secret_version(
            request={'name': secret_name}
        ).payload.data.decode('UTF-8')

        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=password
        )
        cursor = conn.cursor()

        # Get weekly median RUL for last 4 weeks
        cursor.execute("""
            SELECT
                DATE_TRUNC('week', timestamp) AS week,
                PERCENTILE_CONT(0.5)
                    WITHIN GROUP (ORDER BY predicted_rul) AS median_rul,
                COUNT(*) AS row_count
            FROM predictions
            WHERE timestamp > NOW() - INTERVAL '4 weeks'
            GROUP BY week
            ORDER BY week
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(rows) < 2:
            result['message'] = (
                f'Not enough weekly data for drift check ({len(rows)} weeks). '
                'Need at least 2 weeks of predictions.'
            )
            logging.info(result['message'])
            return result

        # Compute 4-week average and compare to current week
        medians      = [float(r[1]) for r in rows]
        avg_4week    = sum(medians[:-1]) / len(medians[:-1])
        current_wk   = medians[-1]

        result['current_median'] = round(current_wk, 4)
        result['4_week_avg']     = round(avg_4week, 4)

        if avg_4week > 0:
            drop_pct = (avg_4week - current_wk) / avg_4week
            result['drop_pct'] = round(drop_pct, 4)

            if drop_pct > 0.20:
                result['prediction_drift'] = True
                result['message'] = (
                    f'PREDICTION DRIFT DETECTED: current median RUL {current_wk:.2f} '
                    f'is {drop_pct*100:.1f}% below 4-week avg {avg_4week:.2f}. '
                    'Triggering immediate retraining.'
                )
                logging.warning(result['message'])
            else:
                result['message'] = (
                    f'Prediction drift OK: current median {current_wk:.2f}, '
                    f'4-week avg {avg_4week:.2f} (drop={drop_pct*100:.1f}%)'
                )
                logging.info(result['message'])
        else:
            result['message'] = '4-week average RUL is zero — cannot compute drift.'
            logging.warning(result['message'])

    except ImportError as e:
        result['message'] = f'Required package not installed: {e}'
        logging.warning(result['message'])
    except Exception as e:
        result['message'] = f'Prediction drift check failed (non-fatal): {e}'
        logging.warning(result['message'])

    return result
