using UnityEngine;
using UnityEngine.InputSystem;

public class RocketCamera : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("Drag your ROOT rocket GameObject here.")]
    public Transform target;

    [Header("Camera Settings")]
    [Tooltip("Fixed distance from rocket centre in metres.")]
    public float distance = 30f;

    [Tooltip("Fixed height angle above rocket in degrees.")]
    public float verticalAngle = 15f;

    [Tooltip("Fixed horizontal angle around rocket in degrees.")]
    public float horizontalAngle = 0f;

    [Tooltip("How smoothly camera moves to position.")]
    public float smoothSpeed = 8f;

    private Vector3 currentVelocity = Vector3.zero;

    void LateUpdate()
    {
        if (target == null) return;

        // Fixed position on a sphere around the rocket
        // Camera does NOT rotate with the rocket
        // So if rocket spins you will see it spin
        Quaternion fixedRotation = Quaternion.Euler(verticalAngle, horizontalAngle, 0f);
        Vector3 offset = fixedRotation * new Vector3(0f, 0f, -distance);
        Vector3 desiredPosition = target.position + offset;

        // Smooth follow
        transform.position = Vector3.SmoothDamp(
            transform.position,
            desiredPosition,
            ref currentVelocity,
            1f / smoothSpeed
        );

        // Always look at rocket centre
        // Camera orientation is WORLD fixed — not rocket fixed
        // This means you will see the rocket rotate/spin clearly
        transform.LookAt(target.position);
    }
}