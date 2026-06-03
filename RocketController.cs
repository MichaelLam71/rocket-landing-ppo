using UnityEngine;

public class RocketController : MonoBehaviour
{
    private Rigidbody rb;

    [Header("Engine")]
    public float maxThrust = 367000f;
    public float rcsForce = 400000f;

    [Header("Spawn")]
    public float spawnHeight = 30f;
    public float spawnPosRange = 0f;
    public float spawnAngleRange = 5f;
    public float spawnVelocityRange = 0f;

    [Header("Physics")]
    public Vector3 centerOfMass = new Vector3(0f, 0.3f, 0f);

    [Header("Debug")]
    public bool manualControl = false;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.centerOfMass = centerOfMass;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        ResetRocket();
    }

    public void ApplyAction(float thrust, float rcsX, float rcsZ)
    {
        // RCS attitude control (works even without main engine)
        rb.AddTorque(new Vector3(rcsX * rcsForce, 0f, rcsZ * rcsForce), ForceMode.Force);

        // main engine thrust (straight up)
        if (thrust > 0.01f)
        {
            rb.AddForce(transform.up * thrust * maxThrust, ForceMode.Force);
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
    }

    void Update()
    {
        if (!manualControl) return;

        float thrust = Input.GetKey(KeyCode.Space) ? 1f : 0f;
        float rcsX = 0f, rcsZ = 0f;

        if (Input.GetKey(KeyCode.W)) rcsX = -1f;
        if (Input.GetKey(KeyCode.S)) rcsX = 1f;
        if (Input.GetKey(KeyCode.A)) rcsZ = 1f;
        if (Input.GetKey(KeyCode.D)) rcsZ = -1f;

        ApplyAction(thrust, rcsX, rcsZ);
    }

    public Rigidbody Rb => rb;
    public float Mass => rb.mass;
}