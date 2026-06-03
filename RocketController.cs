using UnityEngine;

public class RocketController : MonoBehaviour
{
    private Rigidbody rb;

    [Header("References")]
    public Transform enginePivot;

    [Header("Engine")]
    public float maxThrust = 800f;
    public float gimbalAngle = 20f;
    public float gimbalSmoothSpeed = 10f;

    [Header("Spawn")]
    public float spawnHeight = 5f;
    public float spawnPosRange = 0f;
    public float spawnAngleRange = 0f;
    public float spawnVelocityRange = 0f;

    [Header("Physics")]
    public Vector3 centerOfMass = new Vector3(0f, 0.3f, 0f);

    private float currentGimbalX = 0f;
    private float currentGimbalZ = 0f;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();

        if (enginePivot == null)
        {
            GameObject pivot = new GameObject("EnginePivot");
            pivot.transform.SetParent(transform);
            pivot.transform.localPosition = new Vector3(0f, -1f, 0f);
            pivot.transform.localRotation = Quaternion.identity;
            enginePivot = pivot.transform;
            Debug.Log("Auto-created EnginePivot");
        }

        rb.centerOfMass = centerOfMass;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        ResetRocket();
    }

    public void ApplyAction(float thrust, float gimbalX, float gimbalZ)
    {
        currentGimbalX = Mathf.Lerp(currentGimbalX, gimbalX * gimbalAngle,
            Time.fixedDeltaTime * gimbalSmoothSpeed);
        currentGimbalZ = Mathf.Lerp(currentGimbalZ, gimbalZ * gimbalAngle,
            Time.fixedDeltaTime * gimbalSmoothSpeed);

        enginePivot.localRotation = Quaternion.Euler(currentGimbalX, 0f, currentGimbalZ);

        if (thrust > 0.01f)
        {
            rb.AddForceAtPosition(
                enginePivot.up * thrust * maxThrust,
                enginePivot.position,
                ForceMode.Force
            );
        }
    }

    public void ResetRocket()
    {
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        float x = Random.Range(-spawnPosRange, spawnPosRange);
        float z = Random.Range(-spawnPosRange, spawnPosRange);
        transform.position = new Vector3(x, spawnHeight, z);

        float pitch = Random.Range(-spawnAngleRange, spawnAngleRange);
        float roll  = Random.Range(-spawnAngleRange, spawnAngleRange);
        transform.rotation = Quaternion.Euler(pitch, 0f, roll);

        if (spawnVelocityRange > 0f)
        {
            rb.linearVelocity = new Vector3(
                Random.Range(-spawnVelocityRange, spawnVelocityRange),
                Random.Range(-spawnVelocityRange * 2f, 0f),
                Random.Range(-spawnVelocityRange, spawnVelocityRange));
            rb.angularVelocity = spawnVelocityRange * new Vector3(
                Random.Range(-0.5f, 0.5f),
                Random.Range(-0.5f, 0.5f),
                Random.Range(-0.5f, 0.5f));
        }

        currentGimbalX = 0f;
        currentGimbalZ = 0f;
        enginePivot.localRotation = Quaternion.identity;
    }

    // public getters so PythonBridge (or a PID controller) can read physics
    public Rigidbody Rb => rb;
    public float Mass => rb.mass;
}