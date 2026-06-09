using UnityEngine;

public class ColdGasThruster : MonoBehaviour
{
    [Header("Thruster Settings")]
    [Tooltip("Force this thruster produces in Newtons.")]
    public float thrustForce = 400f;

    [Header("Direction")]
    [Tooltip("Local axis this thruster fires along.")]
    public Vector3 thrustDirection = Vector3.up;

    private Rigidbody rb;
    private bool isFiring = false;

    void Start()
    {
        rb = GetComponentInParent<Rigidbody>();

        if (rb == null)
        {
            Debug.LogError("ColdGasThruster on " + gameObject.name +
                           ": No Rigidbody found in parent!");
        }
    }

    void FixedUpdate()
    {
        if (isFiring && rb != null)
        {
            Vector3 worldDir = transform.TransformDirection(thrustDirection.normalized);
            rb.AddForceAtPosition(worldDir * thrustForce, transform.position, ForceMode.Force);
        }
    }

    public void Fire()
    {
        isFiring = true;
    }

    public void StopFiring()
    {
        isFiring = false;
    }

    public bool IsFiring()
    {
        return isFiring;
    }
}