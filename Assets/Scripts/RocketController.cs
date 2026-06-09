using UnityEngine;

public class RocketController : MonoBehaviour
{
    private Rigidbody rb;

    [Header("Engine")]
    public float maxThrust = 470880f;
    public float rcsForce = 500000f;

    [Header("Fuel")]
    public float dryMass = 22000f;
    public float fuelMass = 2000f;
    public float specificImpulse = 282f;
    private float initialFuelMass;

    [Header("Aerodynamics")]
    public bool useAirDrag = true;
    public float airDensity = 1.225f;
    public float dragCoefficient = 0.30f;
    public float frontalArea = 10.8f;

    [Header("Spawn")]
    public float spawnHeight = 200f;
    public float spawnPosRange = 7f;
    public float spawnAngleRange = 35f;
    public float spawnVelocityRange = 0f;

    [Header("Physics")]
    public Vector3 centerOfMass = new Vector3(0f, 0.3f, 0f);

    [Header("Effects")]
    public ParticleSystem engineFlame;

    [Header("Debug")]
    public bool manualControl = false;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.centerOfMass = centerOfMass;
        
        rb.inertiaTensor = new Vector3(500000f, 500000f, 500000f);
        rb.inertiaTensorRotation = Quaternion.identity;
        
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        initialFuelMass = fuelMass;    
        UpdateMass();                   
        ResetRocket();                  
    }

    void FixedUpdate()
    {
        
        Vector3 angVel = rb.angularVelocity;
        angVel.y *= 0.95f;  // dampen 5% per frame
        rb.angularVelocity = angVel;
        ApplyAirDrag();

        if (manualControl)
        {
            float thrust = Input.GetKey(KeyCode.Space) ? 1f : 0f;
            float rcsX = 0f, rcsZ = 0f;

            if (Input.GetKey(KeyCode.W)) rcsX = -1f;
            if (Input.GetKey(KeyCode.S)) rcsX = 1f;
            if (Input.GetKey(KeyCode.A)) rcsZ = 1f;
            if (Input.GetKey(KeyCode.D)) rcsZ = -1f;

            ApplyAction(thrust, rcsX, rcsZ);
        }
    }

    public void ApplyAction(float thrust, float rcsX, float rcsZ)
    {
        // RCS attitude control (works even without main engine)
        rb.AddTorque(new Vector3(rcsX * rcsForce, 0f, rcsZ * rcsForce), ForceMode.Force);

        // main engine thrust (straight up)
        if (thrust > 0.01f && fuelMass > 0f)
        {
            rb.AddForce(transform.up * thrust * maxThrust, ForceMode.Force);
            ConsumeFuel(thrust * maxThrust);
            ControlFlame(true);
        }
        else
        {
            ControlFlame(false);
        }
    }

    void ConsumeFuel(float thrust)
    {
        float massFlowRate = thrust / (specificImpulse * 9.81f);
        fuelMass -= massFlowRate * Time.fixedDeltaTime;
        if (fuelMass < 0f) fuelMass = 0f;
        UpdateMass();
    }

    void UpdateMass()
    {
        rb.mass = dryMass + fuelMass;
    }

    void ApplyAirDrag()
    {
        if (!useAirDrag) return;

        Vector3 velocity = rb.linearVelocity;
        float speed = velocity.magnitude;
        if (speed < 0.01f) return;

        float dragMagnitude = 0.5f * airDensity * dragCoefficient * frontalArea * speed * speed;
        rb.AddForce(-velocity.normalized * dragMagnitude, ForceMode.Force);
    }

    void ControlFlame(bool thrusting)
    {
        if (engineFlame == null) return;
        if (thrusting && !engineFlame.isPlaying) engineFlame.Play();
        if (!thrusting && engineFlame.isPlaying) engineFlame.Stop();
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

        // reset fuel
        fuelMass = initialFuelMass;
        UpdateMass();
        ControlFlame(false);
    }

    public Rigidbody Rb => rb;
    public float Mass => rb.mass;

    
}